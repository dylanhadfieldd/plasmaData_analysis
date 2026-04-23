from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

NIST_ENDPOINT = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
NIST_TIMEOUT_S = 45
ROMAN_ION_RE = re.compile(r"^\s*([A-Za-z]{1,3})\s+([IVX]+)\s*$")

DEFAULT_NIST_SPECTRA = [
    "H I",
    "N I",
    "N II",
    "O I",
    "O II",
    "Ar I",
    "Ar II",
    "He I",
    "He II",
    "C I",
]

DEFAULT_TARGET_SPECIES = [
    {"wavelength_nm": 282.0, "species": "OH"},
    {"wavelength_nm": 308.0, "species": "OH"},
    {"wavelength_nm": 315.0, "species": "N2"},
    {"wavelength_nm": 328.0, "species": "N2+"},
    {"wavelength_nm": 336.0, "species": "N2"},
    {"wavelength_nm": 356.0, "species": "N2"},
    {"wavelength_nm": 374.0, "species": "N2"},
    {"wavelength_nm": 379.0, "species": "N2"},
    {"wavelength_nm": 390.0, "species": "N2+"},
    {"wavelength_nm": 401.0, "species": "N2"},
    {"wavelength_nm": 405.0, "species": "N2"},
]


def normalize_nist_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.startswith("="):
        s = s[1:]
    s = s.strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    s = s.replace('""', '"').strip().strip('"').strip()
    return s


def normalize_nist_lines(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    nist = df.copy()
    nist.columns = [str(c).strip() for c in nist.columns]
    if "wavelength_nm" not in nist.columns:
        raise ValueError("NIST data must include column: wavelength_nm")

    nist["wavelength_nm"] = pd.to_numeric(nist["wavelength_nm"], errors="coerce")
    nist = nist.dropna(subset=["wavelength_nm"]).reset_index(drop=True)

    if "rel_intensity" in nist.columns:
        nist["rel_intensity"] = pd.to_numeric(nist["rel_intensity"], errors="coerce")
    elif "aki_s-1" in nist.columns:
        nist["rel_intensity"] = pd.to_numeric(nist["aki_s-1"], errors="coerce")

    if "species" not in nist.columns:
        if "spectra_query" in nist.columns:
            nist["species"] = nist["spectra_query"]
        else:
            nist["species"] = "unknown"

    if "element" not in nist.columns:
        nist["element"] = ""
    if "ion_stage" not in nist.columns:
        nist["ion_stage"] = ""
    if "transition" not in nist.columns:
        nist["transition"] = nist.get("transition_type", "")

    nist["source"] = source_label
    return nist


def load_fetch_spectra(path: Path) -> List[str]:
    if not path.exists():
        return DEFAULT_NIST_SPECTRA

    df = pd.read_csv(path)
    if df.empty:
        return DEFAULT_NIST_SPECTRA

    if "spectra" in df.columns:
        values = df["spectra"].tolist()
    else:
        values = df.iloc[:, 0].tolist()

    spectra = []
    seen = set()
    for raw in values:
        sp = str(raw).strip()
        if not sp or sp.lower() == "nan":
            continue
        if sp in seen:
            continue
        seen.add(sp)
        spectra.append(sp)
    return spectra or DEFAULT_NIST_SPECTRA


def line_column(df: pd.DataFrame, prefix: str) -> str | None:
    low = {str(c).lower(): str(c) for c in df.columns}
    for c_low, c in low.items():
        if c_low.startswith(prefix):
            return c
    return None


def parse_nist_csv_text(text: str, spectra: str) -> pd.DataFrame:
    if text.lstrip().startswith("<!DOCTYPE html") or "<title>NIST ASD : Input Error</title>" in text:
        msg = "NIST returned an HTML error page"
        if "Error Message:" in text:
            snippet = text.split("Error Message:", 1)[1]
            snippet = re.sub(r"<[^>]+>", " ", snippet)
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if snippet:
                msg = f"NIST input error: {snippet[:220]}"
        raise ValueError(msg)

    raw_df = pd.read_csv(io.StringIO(text), dtype=str)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    raw_df = raw_df[[c for c in raw_df.columns if not c.lower().startswith("unnamed:")]]
    if raw_df.empty:
        return pd.DataFrame(
            columns=["wavelength_nm", "obs_wl_nm", "ritz_wl_nm", "aki_s-1", "acc", "transition_type", "spectra_query"]
        )

    cleaned = raw_df.apply(lambda col: col.map(normalize_nist_value))
    obs_col = line_column(cleaned, "obs_wl")
    ritz_col = line_column(cleaned, "ritz_wl")
    aki_col = line_column(cleaned, "aki")
    acc_col = "Acc" if "Acc" in cleaned.columns else line_column(cleaned, "acc")
    type_col = "Type" if "Type" in cleaned.columns else line_column(cleaned, "type")

    if obs_col is None and ritz_col is None:
        raise ValueError("NIST CSV missing observed/Ritz wavelength columns")

    out = pd.DataFrame()
    out["obs_wl_nm"] = pd.to_numeric(cleaned[obs_col], errors="coerce") if obs_col else np.nan
    out["ritz_wl_nm"] = pd.to_numeric(cleaned[ritz_col], errors="coerce") if ritz_col else np.nan
    out["wavelength_nm"] = out["obs_wl_nm"].fillna(out["ritz_wl_nm"])
    out["aki_s-1"] = pd.to_numeric(cleaned[aki_col], errors="coerce") if aki_col else np.nan
    out["acc"] = cleaned[acc_col].where(cleaned[acc_col] != "", np.nan) if acc_col else np.nan
    out["transition_type"] = cleaned[type_col].where(cleaned[type_col] != "", np.nan) if type_col else np.nan
    out["spectra_query"] = spectra

    elem = ""
    ion = ""
    match = ROMAN_ION_RE.match(spectra)
    if match:
        elem = match.group(1)
        ion = match.group(2)
    out["element"] = elem
    out["ion_stage"] = ion
    out["species"] = spectra
    out["transition"] = out["transition_type"]
    out["rel_intensity"] = out["aki_s-1"]
    out = out.dropna(subset=["wavelength_nm"]).reset_index(drop=True)
    return out


def fetch_nist_lines_live(low_nm: float, high_nm: float, spectra_list: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    status_rows: List[Dict[str, object]] = []

    for spectra in spectra_list:
        params = {
            "spectra": str(spectra),
            "limits_type": "0",
            "low_w": f"{low_nm:.3f}",
            "upp_w": f"{high_nm:.3f}",
            "unit": "1",
            "de": "0",
            "format": "2",
            "line_out": "0",
            "en_unit": "0",
            "output": "0",
            "page_size": "15",
            "show_obs_wl": "1",
            "show_calc_wl": "1",
            "unc_out": "1",
            "show_av": "2",
            "allowed_out": "1",
            "forbid_out": "1",
            "remove_js": "on",
            "submit": "Retrieve Data",
        }
        url = f"{NIST_ENDPOINT}?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "plasmaData_analysis/1.0 (+https://physics.nist.gov/)"})

        try:
            with urlopen(req, timeout=NIST_TIMEOUT_S) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            df = parse_nist_csv_text(text, str(spectra))
            frames.append(df)
            status_rows.append(
                {"spectra": spectra, "status": "ok", "line_count": int(len(df)), "message": "", "source_url": url}
            )
        except (HTTPError, URLError, TimeoutError, ValueError, pd.errors.ParserError) as e:
            status_rows.append(
                {"spectra": spectra, "status": "fail", "line_count": 0, "message": str(e), "source_url": url}
            )

    status_df = pd.DataFrame(status_rows)
    if not frames:
        return pd.DataFrame(), status_df

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(
        subset=["spectra_query", "wavelength_nm", "obs_wl_nm", "ritz_wl_nm", "transition_type"], keep="first"
    ).reset_index(drop=True)
    return out, status_df


def load_local_nist_lines(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Local fallback file not found: {path}")
        return None

    nist = pd.read_csv(path)
    return normalize_nist_lines(nist, source_label="local_csv")


def get_nist_lines_for_range(
    low_nm: float,
    high_nm: float,
    fetch_species_csv: Path,
    fallback_csv: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    spectra_list = load_fetch_spectra(fetch_species_csv)
    print(
        f"[INFO] Fetching NIST lines from {NIST_ENDPOINT} for range {low_nm:.1f}-{high_nm:.1f} nm "
        f"across {len(spectra_list)} spectra."
    )

    live_df, status_df = fetch_nist_lines_live(low_nm=low_nm, high_nm=high_nm, spectra_list=spectra_list)
    if not live_df.empty:
        normalized = normalize_nist_lines(live_df, source_label="live_nist")
        ok = int((status_df["status"] == "ok").sum()) if "status" in status_df.columns else 0
        fail = int((status_df["status"] != "ok").sum()) if "status" in status_df.columns else 0
        print(f"[OK] Pulled {len(normalized)} NIST lines (queries ok={ok}, fail={fail}).")
        return normalized, status_df

    print(f"[WARN] Live NIST fetch returned no usable lines. Falling back to {fallback_csv} if available.")
    try:
        local = load_local_nist_lines(fallback_csv)
    except Exception as e:
        print(f"[FAIL] Could not load fallback {fallback_csv}: {e}")
        return None, status_df
    return local, status_df


def load_target_species_lines(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(DEFAULT_TARGET_SPECIES)

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(DEFAULT_TARGET_SPECIES)

    columns_low = {str(c).strip().lower(): str(c) for c in df.columns}
    wl_col = columns_low.get("wavelength_nm") or columns_low.get("wavelength(nm)") or columns_low.get("wavelength")
    sp_col = columns_low.get("species") or columns_low.get("reactive species") or columns_low.get("reactive_species")
    if wl_col is None or sp_col is None:
        raise ValueError(
            f"{path} must include wavelength and species columns. "
            "Accepted names include wavelength_nm / wavelength(nm) and species / reactive species."
        )

    out = pd.DataFrame(
        {
            "wavelength_nm": pd.to_numeric(df[wl_col], errors="coerce"),
            "species": df[sp_col].astype(str).str.strip(),
        }
    )
    out = out.dropna(subset=["wavelength_nm"])
    out = out[out["species"] != ""]
    out = out.drop_duplicates(subset=["wavelength_nm", "species"], keep="first").reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(DEFAULT_TARGET_SPECIES)
    return out.sort_values(["wavelength_nm", "species"], ignore_index=True)

