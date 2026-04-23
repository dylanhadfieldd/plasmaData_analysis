from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

TRIAL_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.(?P<trial>\d+)$")
SAFE_TEXT_RE = re.compile(r"[^a-z0-9]+")
INPUT_DIRS = [Path("data/air"), Path("data/diameter")]
WAVELENGTH_ROUND = 3


def split_base_and_trial(stem: str) -> Tuple[str, Optional[int]]:
    match = TRIAL_SUFFIX_RE.match(stem)
    if not match:
        return stem, None
    return match.group("base"), int(match.group("trial"))


def slug(text: str) -> str:
    out = SAFE_TEXT_RE.sub("_", text.strip().lower()).strip("_")
    return out or "signal"


def find_input_files() -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for data_dir in INPUT_DIRS:
        dataset = data_dir.name.lower()
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.glob("*.csv")):
            files.append((dataset, path))
    return files


def parse_air_file(path: Path) -> List[Tuple[str, Dict[str, str], pd.DataFrame]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx: Optional[int] = None

    for i, line in enumerate(lines):
        s = line.strip().lower()
        if not s:
            continue
        if s.startswith("wavelength") and ("," in s or "\t" in s):
            header_idx = i
            break
        if "wavelength" in s and "irradiance" in s and ("," in s or "\t" in s):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find spectrum header in {path}")

    meta: Dict[str, str] = {}
    for line in lines[:header_idx]:
        line = line.strip()
        if not line or "," not in line:
            continue
        k, v = line.split(",", 1)
        k = k.strip().lower()
        if k:
            meta[k] = v.strip()

    df = pd.read_csv(
        pd.io.common.StringIO("\n".join(lines[header_idx:])),
        sep=",",
        engine="python",
        skip_blank_lines=True,
    )
    df.columns = [c.strip().lower() for c in df.columns]

    wl_col = next((c for c in df.columns if "wavelength" in c), None)
    ir_col = next((c for c in df.columns if "irradiance" in c), None)
    if wl_col is not None and ir_col is None:
        ir_col = next((c for c in df.columns if c != wl_col), None)
    if wl_col is None or ir_col is None:
        raise ValueError(f"Could not identify wavelength/irradiance columns in {path}: {df.columns.tolist()}")

    out = df[[wl_col, ir_col]].copy()
    out.columns = ["wavelength_nm", "irradiance_W_m2_nm"]
    out["wavelength_nm"] = pd.to_numeric(out["wavelength_nm"], errors="coerce")
    out["irradiance_W_m2_nm"] = pd.to_numeric(out["irradiance_W_m2_nm"], errors="coerce")
    out = out.dropna(subset=["wavelength_nm", "irradiance_W_m2_nm"]).sort_values("wavelength_nm")
    return [("bulk", meta, out)]


def parse_diameter_file(path: Path) -> List[Tuple[str, Dict[str, str], pd.DataFrame]]:
    df = pd.read_csv(path, header=1)
    df.columns = [c.strip() for c in df.columns]

    wl_col = next((c for c in df.columns if "wavelength" in c.lower()), None)
    if wl_col is None:
        raise ValueError(f"Could not find wavelength column in {path}: {df.columns.tolist()}")

    wl = pd.to_numeric(df[wl_col], errors="coerce")
    curves: List[Tuple[str, Dict[str, str], pd.DataFrame]] = []

    for col in df.columns:
        if col == wl_col:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        out = pd.DataFrame(
            {
                "wavelength_nm": wl,
                "irradiance_W_m2_nm": y,
            }
        )
        out = out.dropna(subset=["wavelength_nm", "irradiance_W_m2_nm"]).sort_values("wavelength_nm")
        if len(out) < 2:
            continue
        curves.append((col.strip(), {}, out))

    if not curves:
        raise ValueError(f"No usable diameter channels found in {path}")
    return curves


def parse_spectrum_file(dataset: str, path: Path) -> List[Tuple[str, Dict[str, str], pd.DataFrame]]:
    if str(dataset).strip().lower() == "diameter":
        return parse_diameter_file(path)
    return parse_air_file(path)

