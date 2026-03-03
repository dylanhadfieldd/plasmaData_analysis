#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.plot_style import apply_publication_style, get_palette, spectral_interval_label, style_axes

INPUT_LONG = Path("output/spectra_long.csv")
OUTPUT_ROOT = Path("output")
WAVELENGTH_ROUND = 3
DPI = 200
SHOW_STD_BAND = True
NORMALIZE_Y = True
SAFE_TEXT_RE = re.compile(r"[^a-z0-9]+")
AIR_BASELINE_PARAM = "100H"
AIR_COMPARE_PARAMS = ("5H..01A", "5H..5A", "5H..9A")


def safe_name(text: str) -> str:
    out = SAFE_TEXT_RE.sub("_", text.strip().lower()).strip("_")
    return out or "plot"


def average_group(dataset: str, param_set: str, channel: str, group: pd.DataFrame) -> pd.DataFrame:
    series_list: List[pd.Series] = []
    for sample_id, g in group.groupby("sample_id", dropna=False):
        wl = np.round(g["wavelength_nm"].to_numpy(dtype=float), WAVELENGTH_ROUND)
        y = g["irradiance_W_m2_nm"].to_numpy(dtype=float)
        s = pd.Series(y, index=wl, name=str(sample_id))
        series_list.append(s[~pd.Index(s.index).duplicated(keep="first")])

    if not series_list:
        raise ValueError("No curves in group")

    aligned = pd.concat(series_list, axis=1, sort=True)
    mean = aligned.mean(axis=1, skipna=True)
    std = aligned.std(axis=1, ddof=1, skipna=True)
    n = aligned.count(axis=1)
    return pd.DataFrame(
        {
            "dataset": dataset,
            "param_set": param_set,
            "channel": channel,
            "wavelength_nm": mean.index.to_numpy(dtype=float),
            "irradiance_mean": mean.to_numpy(dtype=float),
            "irradiance_std": std.to_numpy(dtype=float),
            "n_curves": n.to_numpy(dtype=int),
        }
    ).sort_values("wavelength_nm", ignore_index=True)


def curve_label(row: pd.Series) -> str:
    channel = str(row.get("channel", "")).strip()
    label = f"{row['dataset']}:{row['param_set']}"
    if channel and channel.lower() not in {"bulk", "irradiance"}:
        label = f"{label}:{channel}"
    return label


def plot_tables(tables: List[pd.DataFrame], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = get_palette(len(tables))
    x_all: List[np.ndarray] = []
    for color, t in zip(palette, tables):
        row = t.iloc[0]
        x = t["wavelength_nm"].to_numpy(dtype=float)
        x_all.append(x)
        y = t["irradiance_mean"].to_numpy(dtype=float)
        s = t["irradiance_std"].to_numpy(dtype=float)
        if NORMALIZE_Y:
            scale = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
            if scale > 0:
                y = y / scale
                s = s / scale
            else:
                y = np.zeros_like(y, dtype=float)
                s = np.zeros_like(s, dtype=float)
        ax.plot(x, y, label=curve_label(row), color=color, alpha=0.95)
        if SHOW_STD_BAND:
            n = t["n_curves"].to_numpy()
            mask = np.isfinite(s) & (n >= 2)
            if mask.any():
                ax.fill_between(x[mask], (y - s)[mask], (y + s)[mask], color=color, alpha=0.15)

    ax.set_title(title)
    x_full = np.concatenate(x_all) if x_all else np.array([], dtype=float)
    ax.set_xlabel(f"Wavelength {spectral_interval_label(x_full)}")
    ax.set_ylabel("Normalized Irradiance (a.u.)" if NORMALIZE_Y else "Irradiance (W m$^{-2}$ nm$^{-1}$)")
    style_axes(ax, grid_axis="both")
    if NORMALIZE_Y:
        ax.set_ylim(0, 1.05)
    if tables:
        ax.legend(loc="best", fontsize=8, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_air_pairwise_tables(air_tables: List[pd.DataFrame], out_dir: Path) -> List[Path]:
    by_param: dict[str, pd.DataFrame] = {}
    for t in air_tables:
        param = str(t.iloc[0]["param_set"])
        channel = str(t.iloc[0]["channel"])
        if channel.lower() != "bulk":
            continue
        by_param[param] = t

    baseline = by_param.get(AIR_BASELINE_PARAM)
    if baseline is None:
        return []

    out_paths: List[Path] = []
    for compare_param in AIR_COMPARE_PARAMS:
        other = by_param.get(compare_param)
        if other is None:
            continue
        charts = [baseline, other]
        compare_label = compare_param.replace("5H..", "")
        out_path = out_dir / f"air_100h_vs_{safe_name(compare_label)}.png"
        title = f"Air Averaged Spectra Comparison: {AIR_BASELINE_PARAM} vs {compare_label}"
        plot_tables(charts, out_path, title)
        out_paths.append(out_path)
    return out_paths


def main() -> int:
    apply_publication_style()

    if not INPUT_LONG.exists():
        print(f"Missing {INPUT_LONG}. Run preprocess.py first.")
        return 1

    df = pd.read_csv(INPUT_LONG)
    required = {"dataset", "sample_id", "param_set", "channel", "wavelength_nm", "irradiance_W_m2_nm"}
    if not required.issubset(df.columns):
        print(f"{INPUT_LONG} must have columns: {sorted(required)}")
        return 2

    avg_tables: List[pd.DataFrame] = []
    ok = 0
    bad = 0
    grouped = df.groupby(["dataset", "param_set", "channel"], dropna=False)
    for (dataset, param_set, channel), g in grouped:
        try:
            table = average_group(str(dataset), str(param_set), str(channel), g)
            avg_tables.append(table)
            print(f"[OK] averaged {dataset}/{param_set}/{channel} from {g['sample_id'].nunique()} curve(s)")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {dataset}/{param_set}/{channel}: {e}")
            bad += 1

    if not avg_tables:
        print("No groups successfully averaged.")
        return 2

    meta_compared_dir = OUTPUT_ROOT / "meta" / "spectral" / "base" / "charts" / "compared"
    meta_compared_dir.mkdir(parents=True, exist_ok=True)
    output_csv = meta_compared_dir / "combined_averages_long.csv"
    output_png = meta_compared_dir / "combined.png"

    combined_long = pd.concat(avg_tables, ignore_index=True)
    combined_long.to_csv(output_csv, index=False)
    plot_tables(avg_tables, output_png, "Averaged Spectra by Dataset / Parameter Set / Channel")

    for dataset, ds_table in combined_long.groupby("dataset", dropna=False):
        ds_tables = [t for t in avg_tables if str(t.iloc[0]["dataset"]) == str(dataset)]
        ds_dir = OUTPUT_ROOT / safe_name(str(dataset)) / "spectral" / "base" / "charts" / "compared"
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds_csv = ds_dir / "averages_long.csv"
        ds_png = ds_dir / f"{safe_name(str(dataset))}.png"
        ds_table.to_csv(ds_csv, index=False)
        plot_tables(ds_tables, ds_png, f"Averaged Spectra - {dataset}")
        if str(dataset).lower() == "air":
            pair_paths = plot_air_pairwise_tables(ds_tables, ds_dir)
            for p in pair_paths:
                print(f"  {p}")

    print("\nWrote:")
    print(f"  {output_csv}")
    print(f"  {output_png}")
    for dataset in sorted(combined_long["dataset"].astype(str).unique()):
        ds_dir = OUTPUT_ROOT / safe_name(dataset) / "spectral" / "base" / "charts" / "compared"
        print(f"  {ds_dir / 'averages_long.csv'}")
        print(f"  {ds_dir / (safe_name(dataset) + '.png')}")
    print(f"\nDone. Groups OK={ok} FAIL={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
