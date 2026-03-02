#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

TRIAL_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.(?P<trial>\d+)$")
SAFE_TEXT_RE = re.compile(r"[^a-z0-9]+")
INPUT_DIRS = [Path("data/air"), Path("data/diameter")]
OUTPUT_DIR = Path("output")
WAVELENGTH_ROUND = 3
SCOPES = ("air", "diameter", "meta")


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


def write_outputs(metadata_df: pd.DataFrame, spectra_long_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    wide = spectra_long_df.pivot_table(
        index=["dataset", "sample_id", "param_set", "trial", "channel"],
        columns="wavelength_nm",
        values="irradiance_W_m2_nm",
        aggfunc="first",
    ).reset_index()

    wide.columns = [f"wl_{c:g}" if isinstance(c, (int, float)) else str(c) for c in wide.columns]

    metadata_df.to_csv(out_dir / "metadata.csv", index=False)
    spectra_long_df.to_csv(out_dir / "spectra_long.csv", index=False)
    wide.to_csv(out_dir / "spectra_wide.csv", index=False)


def write_scoped_base_raw(metadata_df: pd.DataFrame, spectra_long_df: pd.DataFrame, out_root: Path) -> None:
    for scope in SCOPES:
        raw_dir = out_root / scope / "spectral" / "base" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if scope == "meta":
            m_part = metadata_df.copy()
            s_part = spectra_long_df.copy()
        else:
            m_part = metadata_df[metadata_df["dataset"].astype(str).str.lower() == scope].copy()
            s_part = spectra_long_df[spectra_long_df["dataset"].astype(str).str.lower() == scope].copy()

        write_outputs(m_part.reset_index(drop=True), s_part.reset_index(drop=True), raw_dir)


def main() -> int:
    in_files = find_input_files()
    if not in_files:
        print(f"No files found under: {', '.join(str(p) for p in INPUT_DIRS)}")
        return 1

    meta_rows: List[Dict[str, object]] = []
    long_rows: List[pd.DataFrame] = []
    ok = 0
    bad = 0

    for dataset, path in in_files:
        try:
            if dataset == "diameter":
                curves = parse_diameter_file(path)
            else:
                curves = parse_air_file(path)

            stem = path.stem
            base, trial = split_base_and_trial(stem)
            curve_count = 0

            for channel, meta, spec in curves:
                channel = channel.strip() or "signal"
                channel_key = slug(channel)
                sample_id = f"{dataset}__{stem}__{channel_key}"

                spec_long = spec.copy()
                spec_long["wavelength_nm"] = spec_long["wavelength_nm"].round(WAVELENGTH_ROUND)
                spec_long.insert(0, "dataset", dataset)
                spec_long.insert(1, "sample_id", sample_id)
                spec_long.insert(2, "param_set", base)
                spec_long.insert(3, "trial", trial)
                spec_long.insert(4, "channel", channel)
                long_rows.append(spec_long)

                meta_row: Dict[str, object] = {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "param_set": base,
                    "trial": trial,
                    "channel": channel,
                    "source_file": str(path.as_posix()),
                }
                for k, v in meta.items():
                    if k in meta_row:
                        meta_row[f"meta_{k}"] = v
                    else:
                        meta_row[k] = v
                meta_rows.append(meta_row)
                curve_count += 1

            print(f"[OK] {path.name} ({dataset}) -> {curve_count} curve(s)")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {path.name} ({dataset}): {e}")
            bad += 1

    if ok == 0 or not long_rows:
        print("No files successfully parsed.")
        return 2

    metadata_df = pd.DataFrame(meta_rows).sort_values(["dataset", "sample_id"], ignore_index=True)
    spectra_long_df = pd.concat(long_rows, ignore_index=True).sort_values(
        ["dataset", "sample_id", "wavelength_nm"], ignore_index=True
    )

    write_outputs(metadata_df, spectra_long_df, OUTPUT_DIR)
    for dataset in sorted(spectra_long_df["dataset"].dropna().astype(str).unique()):
        ds_meta = metadata_df[metadata_df["dataset"] == dataset].reset_index(drop=True)
        ds_long = spectra_long_df[spectra_long_df["dataset"] == dataset].reset_index(drop=True)
        write_outputs(ds_meta, ds_long, OUTPUT_DIR / dataset)
    write_scoped_base_raw(metadata_df, spectra_long_df, OUTPUT_DIR)

    print("\nWrote:")
    print(f"  {OUTPUT_DIR / 'metadata.csv'}")
    print(f"  {OUTPUT_DIR / 'spectra_long.csv'}")
    print(f"  {OUTPUT_DIR / 'spectra_wide.csv'}")
    for dataset in sorted(spectra_long_df["dataset"].dropna().astype(str).unique()):
        print(f"  {OUTPUT_DIR / dataset / 'metadata.csv'}")
        print(f"  {OUTPUT_DIR / dataset / 'spectra_long.csv'}")
        print(f"  {OUTPUT_DIR / dataset / 'spectra_wide.csv'}")
    for scope in SCOPES:
        raw_dir = OUTPUT_DIR / scope / "spectral" / "base" / "raw"
        print(f"  {raw_dir / 'metadata.csv'}")
        print(f"  {raw_dir / 'spectra_long.csv'}")
        print(f"  {raw_dir / 'spectra_wide.csv'}")
    print(f"\nDone. Parsed OK={ok} FAIL={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
