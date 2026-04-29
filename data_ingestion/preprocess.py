#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from analysis.output_paths import SCOPES, ensure_all_scope_layouts, metadata_section_dir
from analysis.scoped_outputs import scoped_slice
from data_ingestion.loading import (
    INPUT_DIRS,
    WAVELENGTH_ROUND,
    find_input_files,
    parse_spectrum_file,
    slug,
    split_base_and_trial,
)


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


def write_scoped_base_raw(metadata_df: pd.DataFrame, spectra_long_df: pd.DataFrame) -> None:
    for scope in SCOPES:
        raw_dir = metadata_section_dir(scope, "spectral")
        raw_dir.mkdir(parents=True, exist_ok=True)
        m_part = scoped_slice(metadata_df, scope, allow_global=True).reset_index(drop=True)
        s_part = scoped_slice(spectra_long_df, scope, allow_global=True).reset_index(drop=True)
        write_outputs(m_part, s_part, raw_dir)


def build_preprocessed_frames() -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    in_files = find_input_files()
    if not in_files:
        raise FileNotFoundError(f"No files found under: {', '.join(str(p) for p in INPUT_DIRS)}")

    meta_rows: List[Dict[str, object]] = []
    long_rows: List[pd.DataFrame] = []
    ok = 0
    bad = 0

    for dataset, path in in_files:
        try:
            curves = parse_spectrum_file(dataset, path)

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
        raise RuntimeError("No files successfully parsed.")

    metadata_df = pd.DataFrame(meta_rows).sort_values(["dataset", "sample_id"], ignore_index=True)
    spectra_long_df = pd.concat(long_rows, ignore_index=True).sort_values(
        ["dataset", "sample_id", "wavelength_nm"], ignore_index=True
    )
    return metadata_df, spectra_long_df, ok, bad


def main() -> int:
    ensure_all_scope_layouts()

    try:
        metadata_df, spectra_long_df, ok, bad = build_preprocessed_frames()
    except FileNotFoundError as e:
        print(str(e))
        return 1
    except RuntimeError as e:
        print(str(e))
        return 2

    write_scoped_base_raw(metadata_df, spectra_long_df)

    print("\nWrote:")
    for scope in SCOPES:
        raw_dir = metadata_section_dir(scope, "spectral")
        print(f"  {raw_dir / 'metadata.csv'}")
        print(f"  {raw_dir / 'spectra_long.csv'}")
        print(f"  {raw_dir / 'spectra_wide.csv'}")
    print(f"\nDone. Parsed OK={ok} FAIL={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
