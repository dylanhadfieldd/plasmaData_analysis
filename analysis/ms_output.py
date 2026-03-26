#!/usr/bin/env python3
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
from analysis.output_paths import SCOPES, ensure_all_scope_layouts, metadata_csv_path, metadata_section_dir

IN_LONG = metadata_csv_path("meta", "spectral", "spectra_long.csv")
NIST_CSV = Path("configs/nist_lines.csv")
NIST_FETCH_SPECIES_CSV = Path("configs/nist_fetch_species.csv")
TARGET_SPECIES_CSV = Path("configs/target_species_lines.csv")
NIST_ENDPOINT = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
NIST_TIMEOUT_S = 45

AVERAGED_CURVES_NAME = "averaged_curves_long.csv"
AVERAGED_PEAKS_NAME = "averaged_peaks_top10.csv"
TRIAL_PEAKS_NAME = "trial_peaks_top10.csv"
NIST_MATCHES_NAME = "nist_matches_top3.csv"
NIST_MATCH_SUMMARY_NAME = "nist_match_summary.csv"
TARGET_MATCHES_NAME = "target_species_peak_matches.csv"
TARGET_MATCH_SUMMARY_NAME = "target_species_match_summary.csv"
NIST_SOURCE_LINES_NAME = "nist_lines_live.csv"
NIST_FETCH_STATUS_NAME = "nist_fetch_status.csv"
WORKBOOK_NAME = "peak_species_review.xlsx"

TOP_N_PEAKS = 10
TOP_NIST_CANDIDATES = 3
PEAK_DECIMALS = 1
NIST_TOLERANCE_NM = 0.5
TARGET_MATCH_TOLERANCE_NM = 2.0
WAVELENGTH_ROUND = 3

REQUIRED_LONG_COLS = {
    "dataset",
    "sample_id",
    "param_set",
    "channel",
    "wavelength_nm",
    "irradiance_W_m2_nm",
}

PEAK_KEY_COLS = ["dataset", "param_set", "channel", "peak_rank", "peak_wavelength_nm_0p1"]
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
ROMAN_ION_RE = re.compile(r"^\s*([A-Za-z]{1,3})\s+([IVX]+)\s*$")


def require_columns(df: pd.DataFrame, cols: Sequence[str], source_name: str) -> bool:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        print(f"{source_name} missing columns: {missing}")
        return False
    return True


def scope_raw_dir(scope: str) -> Path:
    return metadata_section_dir(scope, "spectral")


def write_scoped_csv(df: pd.DataFrame, file_name: str, allow_global: bool = False) -> List[Path]:
    written: List[Path] = []
    for scope in SCOPES:
        out_dir = scope_raw_dir(scope)
        out_dir.mkdir(parents=True, exist_ok=True)
        if scope == "meta":
            part = df.copy()
        elif allow_global:
            part = df.copy()
        elif "dataset" in df.columns:
            part = df[df["dataset"].astype(str).str.lower() == scope].copy()
        else:
            part = pd.DataFrame(columns=df.columns)
        out_path = out_dir / file_name
        part.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def meta_raw_path(file_name: str) -> Path:
    out = scope_raw_dir("meta") / file_name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def average_curves(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    grouped = df.groupby(["dataset", "param_set", "channel"], dropna=False)
    for (dataset, param_set, channel), group in grouped:
        series_list: List[pd.Series] = []
        for sample_id, g in group.groupby("sample_id", dropna=False):
            wl = np.round(g["wavelength_nm"].to_numpy(dtype=float), WAVELENGTH_ROUND)
            y = g["irradiance_W_m2_nm"].to_numpy(dtype=float)
            s = pd.Series(y, index=wl, name=str(sample_id))
            series_list.append(s[~pd.Index(s.index).duplicated(keep="first")])

        if not series_list:
            continue

        aligned = pd.concat(series_list, axis=1, sort=True)
        mean = aligned.mean(axis=1, skipna=True)
        std = aligned.std(axis=1, ddof=1, skipna=True)
        n_curves = aligned.count(axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": str(dataset),
                    "param_set": str(param_set),
                    "channel": str(channel),
                    "wavelength_nm": mean.index.to_numpy(dtype=float),
                    "irradiance_mean": mean.to_numpy(dtype=float),
                    "irradiance_std": std.to_numpy(dtype=float),
                    "n_curves": n_curves.to_numpy(dtype=int),
                }
            ).sort_values("wavelength_nm", ignore_index=True)
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "param_set",
                "channel",
                "wavelength_nm",
                "irradiance_mean",
                "irradiance_std",
                "n_curves",
            ]
        )

    return pd.concat(rows, ignore_index=True).sort_values(
        ["dataset", "param_set", "channel", "wavelength_nm"], ignore_index=True
    )


def local_maxima_indices(y: np.ndarray) -> List[int]:
    if y.size < 3:
        return []
    out: List[int] = []
    for i in range(1, y.size - 1):
        left = y[i - 1]
        mid = y[i]
        right = y[i + 1]
        if np.isfinite(left) and np.isfinite(mid) and np.isfinite(right) and mid > left and mid > right:
            out.append(i)
    return out


def refine_peak_quadratic(
    x_left: float, y_left: float, x_mid: float, y_mid: float, x_right: float, y_right: float
) -> Dict[str, float]:
    denom = y_left - (2.0 * y_mid) + y_right
    if not np.isfinite(denom) or denom == 0:
        return {"refined_wavelength_nm": float(x_mid), "refined_intensity": float(y_mid)}

    offset = 0.5 * (y_left - y_right) / denom
    if not np.isfinite(offset):
        return {"refined_wavelength_nm": float(x_mid), "refined_intensity": float(y_mid)}

    offset = float(np.clip(offset, -1.0, 1.0))
    step = 0.5 * (x_right - x_left)
    if not np.isfinite(step) or step <= 0:
        return {"refined_wavelength_nm": float(x_mid), "refined_intensity": float(y_mid)}

    refined_wl = float(np.clip(x_mid + (offset * step), min(x_left, x_right), max(x_left, x_right)))
    refined_y = float(y_mid - (0.25 * (y_left - y_right) * offset))
    return {"refined_wavelength_nm": refined_wl, "refined_intensity": refined_y}


def detect_top_peaks(
    wl: np.ndarray, y: np.ndarray, top_n: int, intensity_col_name: str
) -> List[Dict[str, float]]:
    indices = local_maxima_indices(y)
    peaks: List[Dict[str, float]] = []
    for idx in indices:
        refined = refine_peak_quadratic(wl[idx - 1], y[idx - 1], wl[idx], y[idx], wl[idx + 1], y[idx + 1])
        peaks.append(
            {
                "peak_wavelength_nm_grid": float(wl[idx]),
                "peak_wavelength_nm_refined": float(refined["refined_wavelength_nm"]),
                "peak_wavelength_nm_0p1": round(float(refined["refined_wavelength_nm"]), PEAK_DECIMALS),
                intensity_col_name: float(y[idx]),
                "peak_intensity_refined": float(refined["refined_intensity"]),
            }
        )

    peaks.sort(key=lambda row: (-row[intensity_col_name], row["peak_wavelength_nm_grid"]))
    return peaks[:top_n]


def build_peak_table(
    spectra_df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    intensity_label: str,
    top_n: int,
    extra_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    extra_cols = list(extra_cols or [])
    rows: List[Dict[str, object]] = []
    for key, g in spectra_df.groupby(list(group_cols), dropna=False):
        g = g.sort_values("wavelength_nm", ignore_index=True)
        wl = g["wavelength_nm"].to_numpy(dtype=float)
        y = g[value_col].to_numpy(dtype=float)

        peaks = detect_top_peaks(wl, y, top_n=top_n, intensity_col_name=intensity_label)
        if not peaks:
            continue

        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(group_cols, key))

        extra_map: Dict[str, object] = {}
        for col in extra_cols:
            if col in g.columns:
                val = g[col].iloc[0]
                if pd.isna(val):
                    continue
                extra_map[col] = val

        for rank, peak in enumerate(peaks, start=1):
            row: Dict[str, object] = {"peak_rank": rank}
            row.update(key_map)
            row.update(extra_map)
            row.update(peak)
            rows.append(row)

    if not rows:
        cols = list(group_cols) + list(extra_cols) + [
            "peak_rank",
            "peak_wavelength_nm_grid",
            "peak_wavelength_nm_refined",
            "peak_wavelength_nm_0p1",
            intensity_label,
            "peak_intensity_refined",
        ]
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    return out.sort_values(list(group_cols) + ["peak_rank"], ignore_index=True)


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


def match_peaks_to_target_species(
    averaged_peaks: pd.DataFrame, targets: pd.DataFrame, tolerance_nm: float
) -> pd.DataFrame:
    cols = [
        "dataset",
        "param_set",
        "channel",
        "species",
        "target_wavelength_nm",
        "target_rank",
        "matched",
        "matched_peak_rank",
        "matched_peak_wavelength_nm_0p1",
        "matched_peak_intensity",
        "delta_nm",
    ]
    if averaged_peaks.empty or targets.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, object]] = []
    group_cols = ["dataset", "param_set", "channel"]
    required_cols = set(group_cols + ["peak_rank", "peak_wavelength_nm_0p1"])
    if not required_cols.issubset(averaged_peaks.columns):
        return pd.DataFrame(columns=cols)

    for key, gp in averaged_peaks.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        group_map = dict(zip(group_cols, key))
        peak_wl = pd.to_numeric(gp["peak_wavelength_nm_0p1"], errors="coerce")
        peak_rank = pd.to_numeric(gp["peak_rank"], errors="coerce")
        peak_int = pd.to_numeric(
            gp.get("peak_intensity_refined", gp.get("peak_intensity", np.nan)), errors="coerce"
        )

        work = gp.copy()
        work["peak_wl"] = peak_wl
        work["peak_rank_num"] = peak_rank
        work["peak_int_num"] = peak_int
        work = work[np.isfinite(work["peak_wl"])].copy()
        if work.empty:
            continue

        for t_idx, t in targets.reset_index(drop=True).iterrows():
            target_wl = float(t["wavelength_nm"])
            species = str(t["species"])
            delta = np.abs(work["peak_wl"].to_numpy(dtype=float) - target_wl)
            best_pos = int(np.argmin(delta))
            best_row = work.iloc[best_pos]
            best_delta = float(delta[best_pos])
            matched = bool(np.isfinite(best_delta) and best_delta <= float(tolerance_nm))
            rows.append(
                {
                    **group_map,
                    "species": species,
                    "target_wavelength_nm": target_wl,
                    "target_rank": int(t_idx + 1),
                    "matched": matched,
                    "matched_peak_rank": int(best_row["peak_rank_num"]) if matched else np.nan,
                    "matched_peak_wavelength_nm_0p1": float(best_row["peak_wl"]) if matched else np.nan,
                    "matched_peak_intensity": float(best_row["peak_int_num"]) if matched else np.nan,
                    "delta_nm": best_delta,
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(
        ["dataset", "param_set", "channel", "target_rank"], ignore_index=True
    )


def build_target_match_summary(target_matches: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "param_set",
        "channel",
        "species",
        "targets_total",
        "targets_matched",
        "match_rate",
        "mean_delta_nm",
        "mean_matched_peak_intensity",
    ]
    if target_matches.empty:
        return pd.DataFrame(columns=cols)

    d = target_matches.copy()
    d["matched_num"] = d["matched"].astype(bool).astype(int)
    d["delta_for_match"] = pd.to_numeric(d["delta_nm"], errors="coerce").where(d["matched"].astype(bool), np.nan)
    d["int_for_match"] = pd.to_numeric(d["matched_peak_intensity"], errors="coerce").where(
        d["matched"].astype(bool), np.nan
    )

    out = (
        d.groupby(["dataset", "param_set", "channel", "species"], dropna=False)
        .agg(
            targets_total=("target_wavelength_nm", "size"),
            targets_matched=("matched_num", "sum"),
            mean_delta_nm=("delta_for_match", "mean"),
            mean_matched_peak_intensity=("int_for_match", "mean"),
        )
        .reset_index()
    )
    out["match_rate"] = (
        pd.to_numeric(out["targets_matched"], errors="coerce")
        / pd.to_numeric(out["targets_total"], errors="coerce").replace(0, np.nan)
    ).fillna(0.0)
    return out[cols].sort_values(["dataset", "param_set", "channel", "species"], ignore_index=True)


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


def get_nist_lines_for_range(low_nm: float, high_nm: float) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    spectra_list = load_fetch_spectra(NIST_FETCH_SPECIES_CSV)
    print(
        f"[INFO] Fetching NIST lines from {NIST_ENDPOINT} for range {low_nm:.1f}-{high_nm:.1f} nm "
        f"across {len(spectra_list)} spectra."
    )

    live_df, status_df = fetch_nist_lines_live(low_nm=low_nm, high_nm=high_nm, spectra_list=spectra_list)
    if not status_df.empty:
        status_df.to_csv(meta_raw_path(NIST_FETCH_STATUS_NAME), index=False)
    if not live_df.empty:
        normalized = normalize_nist_lines(live_df, source_label="live_nist")
        normalized.to_csv(meta_raw_path(NIST_SOURCE_LINES_NAME), index=False)
        ok = int((status_df["status"] == "ok").sum()) if "status" in status_df.columns else 0
        fail = int((status_df["status"] != "ok").sum()) if "status" in status_df.columns else 0
        print(f"[OK] Pulled {len(normalized)} NIST lines (queries ok={ok}, fail={fail}).")
        return normalized, status_df

    print("[WARN] Live NIST fetch returned no usable lines. Falling back to configs/nist_lines.csv if available.")
    try:
        local = load_local_nist_lines(NIST_CSV)
    except Exception as e:
        print(f"[FAIL] Could not load fallback {NIST_CSV}: {e}")
        return None, status_df
    return local, status_df


def match_peaks_to_nist(averaged_peaks: pd.DataFrame, nist_df: pd.DataFrame | None) -> pd.DataFrame:
    if averaged_peaks.empty or nist_df is None or nist_df.empty:
        base_cols = list(averaged_peaks.columns)
        return pd.DataFrame(columns=base_cols + ["candidate_rank", "delta_nm", "nist_wavelength_nm"])

    rows: List[Dict[str, object]] = []
    has_rel_intensity = "rel_intensity" in nist_df.columns
    pass_through_cols = [c for c in nist_df.columns if c != "wavelength_nm"]

    for peak in averaged_peaks.to_dict(orient="records"):
        peak_wl = float(peak["peak_wavelength_nm_0p1"])
        candidates = nist_df[np.abs(nist_df["wavelength_nm"] - peak_wl) <= NIST_TOLERANCE_NM].copy()
        if candidates.empty:
            continue

        candidates["delta_nm"] = np.abs(candidates["wavelength_nm"] - peak_wl)
        sort_cols = ["delta_nm"]
        asc = [True]
        if has_rel_intensity:
            sort_cols.append("rel_intensity")
            asc.append(False)
        candidates = candidates.sort_values(sort_cols, ascending=asc).head(TOP_NIST_CANDIDATES)

        for rank, (_, cand) in enumerate(candidates.iterrows(), start=1):
            row = dict(peak)
            row["candidate_rank"] = rank
            row["delta_nm"] = float(cand["delta_nm"])
            row["nist_wavelength_nm"] = float(cand["wavelength_nm"])
            for col in pass_through_cols:
                row[f"nist_{col}"] = cand[col]
            rows.append(row)

    if not rows:
        base_cols = list(averaged_peaks.columns)
        return pd.DataFrame(columns=base_cols + ["candidate_rank", "delta_nm", "nist_wavelength_nm"])

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["dataset", "param_set", "channel", "peak_rank", "candidate_rank"], ignore_index=True
    )


def build_nist_match_summary(matches: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "param_set", "channel", "candidate_label", "matched_peak_count", "candidate_rows", "score"]
    if matches.empty:
        return pd.DataFrame(columns=cols)

    label_col = "nist_species" if "nist_species" in matches.columns else None
    if label_col is None and "nist_spectra_query" in matches.columns:
        label_col = "nist_spectra_query"
    if label_col is None and "nist_element" in matches.columns:
        label_col = "nist_element"

    df = matches.copy()
    if label_col is None:
        df["candidate_label"] = "unknown"
    else:
        df["candidate_label"] = df[label_col].fillna("unknown").astype(str)

    df["match_score"] = (1.0 / df["candidate_rank"].clip(lower=1).astype(float)) * (
        1.0 / (1.0 + df["delta_nm"].astype(float))
    )
    key_cols = ["dataset", "param_set", "channel", "candidate_label"]
    grouped = df.groupby(key_cols, dropna=False).agg(
        candidate_rows=("candidate_rank", "size"),
        matched_peak_count=("peak_rank", "nunique"),
        score=("match_score", "sum"),
    )
    out = grouped.reset_index().sort_values(
        ["dataset", "param_set", "channel", "score", "matched_peak_count"],
        ascending=[True, True, True, False, False],
        ignore_index=True,
    )
    return out


def unmatched_averaged_peaks(averaged_peaks: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if averaged_peaks.empty:
        return averaged_peaks.copy()
    if matches.empty:
        return averaged_peaks.copy()

    matched_keys = matches[PEAK_KEY_COLS].drop_duplicates().assign(_matched=1)
    tagged = averaged_peaks.merge(matched_keys, how="left", on=PEAK_KEY_COLS)
    return tagged[tagged["_matched"].isna()].drop(columns=["_matched"]).reset_index(drop=True)


def build_summary(averaged_peaks: pd.DataFrame, trial_peaks: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "param_set", "channel"]

    avg_counts = (
        averaged_peaks.groupby(group_cols, dropna=False).size().rename("n_averaged_peaks").reset_index()
        if not averaged_peaks.empty
        else pd.DataFrame(columns=group_cols + ["n_averaged_peaks"])
    )
    matched_counts = (
        matches[group_cols + ["peak_rank", "peak_wavelength_nm_0p1"]]
        .drop_duplicates()
        .groupby(group_cols, dropna=False)
        .size()
        .rename("n_averaged_peaks_matched")
        .reset_index()
        if not matches.empty
        else pd.DataFrame(columns=group_cols + ["n_averaged_peaks_matched"])
    )
    trial_counts = (
        trial_peaks.groupby(group_cols, dropna=False).size().rename("n_trial_peaks").reset_index()
        if not trial_peaks.empty
        else pd.DataFrame(columns=group_cols + ["n_trial_peaks"])
    )
    candidate_counts = (
        matches.groupby(group_cols, dropna=False).size().rename("n_nist_candidates").reset_index()
        if not matches.empty
        else pd.DataFrame(columns=group_cols + ["n_nist_candidates"])
    )

    summary = avg_counts.merge(matched_counts, on=group_cols, how="outer")
    summary = summary.merge(trial_counts, on=group_cols, how="outer")
    summary = summary.merge(candidate_counts, on=group_cols, how="outer")

    for col in ["n_averaged_peaks", "n_averaged_peaks_matched", "n_trial_peaks", "n_nist_candidates"]:
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0).astype(int)
    summary["n_averaged_peaks_unmatched"] = summary["n_averaged_peaks"] - summary["n_averaged_peaks_matched"]

    return summary.sort_values(group_cols, ignore_index=True)


def write_excel(
    summary: pd.DataFrame,
    averaged_peaks: pd.DataFrame,
    trial_peaks: pd.DataFrame,
    matches: pd.DataFrame,
    nist_match_summary: pd.DataFrame,
    target_matches: pd.DataFrame,
    target_match_summary: pd.DataFrame,
    nist_lines: pd.DataFrame,
    fetch_status: pd.DataFrame,
    unmatched: pd.DataFrame,
    out_path: Path,
) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        averaged_peaks.to_excel(writer, sheet_name="AveragedPeaks", index=False)
        trial_peaks.to_excel(writer, sheet_name="TrialPeaks", index=False)
        matches.to_excel(writer, sheet_name="NISTMatches", index=False)
        nist_match_summary.to_excel(writer, sheet_name="NISTMatchSummary", index=False)
        target_matches.to_excel(writer, sheet_name="TargetSpeciesMatches", index=False)
        target_match_summary.to_excel(writer, sheet_name="TargetSpeciesSummary", index=False)
        nist_lines.to_excel(writer, sheet_name="NISTSourceLines", index=False)
        fetch_status.to_excel(writer, sheet_name="NISTFetchStatus", index=False)
        unmatched.to_excel(writer, sheet_name="UnmatchedAveragedPeaks", index=False)


def main() -> int:
    ensure_all_scope_layouts()

    if not IN_LONG.exists():
        print(f"Missing {IN_LONG}. Run preprocess.py first.")
        return 1

    spectra_long = pd.read_csv(IN_LONG)
    if not require_columns(spectra_long, REQUIRED_LONG_COLS, str(IN_LONG)):
        return 2

    averaged = average_curves(spectra_long)
    if averaged.empty:
        print("No averaged curves could be built from input.")
        return 2

    written_paths: List[Path] = []
    written_paths.extend(write_scoped_csv(averaged, AVERAGED_CURVES_NAME))

    averaged_peaks = build_peak_table(
        averaged,
        group_cols=["dataset", "param_set", "channel"],
        value_col="irradiance_mean",
        intensity_label="peak_intensity",
        top_n=TOP_N_PEAKS,
        extra_cols=["n_curves"],
    )
    written_paths.extend(write_scoped_csv(averaged_peaks, AVERAGED_PEAKS_NAME))

    trial_group_cols = ["dataset", "param_set", "channel", "sample_id"]
    if "trial" in spectra_long.columns:
        trial_group_cols.append("trial")
    trial_peaks = build_peak_table(
        spectra_long,
        group_cols=trial_group_cols,
        value_col="irradiance_W_m2_nm",
        intensity_label="peak_intensity",
        top_n=TOP_N_PEAKS,
    )
    written_paths.extend(write_scoped_csv(trial_peaks, TRIAL_PEAKS_NAME))

    try:
        targets = load_target_species_lines(TARGET_SPECIES_CSV)
    except Exception as e:
        print(f"[FAIL] Could not load target species file {TARGET_SPECIES_CSV}: {e}")
        return 2
    target_matches = match_peaks_to_target_species(
        averaged_peaks=averaged_peaks,
        targets=targets,
        tolerance_nm=TARGET_MATCH_TOLERANCE_NM,
    )
    written_paths.extend(write_scoped_csv(target_matches, TARGET_MATCHES_NAME))
    target_match_summary = build_target_match_summary(target_matches)
    written_paths.extend(write_scoped_csv(target_match_summary, TARGET_MATCH_SUMMARY_NAME))

    range_low = float(np.nanmin(averaged["wavelength_nm"].to_numpy(dtype=float)))
    range_high = float(np.nanmax(averaged["wavelength_nm"].to_numpy(dtype=float)))
    try:
        nist_df, fetch_status = get_nist_lines_for_range(range_low, range_high)
    except Exception as e:
        print(f"[FAIL] Could not retrieve NIST lines: {e}")
        return 2

    matches = match_peaks_to_nist(averaged_peaks, nist_df)
    written_paths.extend(write_scoped_csv(matches, NIST_MATCHES_NAME))
    nist_match_summary = build_nist_match_summary(matches)
    written_paths.extend(write_scoped_csv(nist_match_summary, NIST_MATCH_SUMMARY_NAME))
    written_paths.extend(write_scoped_csv(fetch_status, NIST_FETCH_STATUS_NAME, allow_global=True))
    if nist_df is not None and not nist_df.empty:
        nist_live_path = meta_raw_path(NIST_SOURCE_LINES_NAME)
        nist_df.to_csv(nist_live_path, index=False)
        written_paths.append(nist_live_path)

    unmatched = unmatched_averaged_peaks(averaged_peaks, matches)
    summary = build_summary(averaged_peaks, trial_peaks, matches)

    excel_error: Exception | None = None
    try:
        write_excel(
            summary,
            averaged_peaks,
            trial_peaks,
            matches,
            nist_match_summary,
            target_matches,
            target_match_summary,
            nist_df if nist_df is not None else pd.DataFrame(),
            fetch_status,
            unmatched,
            meta_raw_path(WORKBOOK_NAME),
        )
    except Exception as e:
        excel_error = e

    print("\nWrote:")
    for path in sorted(set(written_paths)):
        print(f"  {path}")
    print(f"  {meta_raw_path(WORKBOOK_NAME)}")

    if excel_error is not None:
        print(f"[FAIL] Excel workbook write failed: {excel_error}")
        return 2

    if nist_df is None:
        print(f"[WARN] NIST matches are empty because live fetch failed and {NIST_CSV} was unavailable/invalid.")
    print(
        f"Done. averaged_curves={len(averaged)} averaged_peaks={len(averaged_peaks)} "
        f"trial_peaks={len(trial_peaks)} nist_matches={len(matches)} "
        f"target_species_matches={len(target_matches)} "
        f"nist_source_lines={0 if nist_df is None else len(nist_df)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
