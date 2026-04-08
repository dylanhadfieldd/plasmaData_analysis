#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.output_paths import (
    ensure_all_scope_layouts,
    metadata_csv_path,
    spectral_compared_dir,
    spectral_composed_dir,
    spectral_individual_dir,
)
from analysis.plot_style import apply_publication_style, get_palette, spectral_interval_label, style_axes

INPUT_LONG = metadata_csv_path("meta", "spectral", "spectra_long.csv")
DPI = 220
NORMALIZE_Y = True
SHOW_STD_BAND = True
WAVELENGTH_ROUND = 3


def normalize_curve(y: np.ndarray) -> np.ndarray:
    if not NORMALIZE_Y:
        return y
    scale = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
    if scale <= 0:
        return np.zeros_like(y, dtype=float)
    return y / scale


def clear_pngs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.png"):
        old.unlink()


def grouped_label(row: pd.Series) -> str:
    channel = str(row.get("channel", "")).strip()
    label = f"{row['dataset']}:{row['param_set']}"
    if channel and channel.lower() not in {"bulk", "irradiance"}:
        label = f"{label}:{channel}"
    return label


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


def plot_tables(tables: List[pd.DataFrame], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = get_palette(len(tables))
    x_all: List[np.ndarray] = []
    for color, t in zip(palette, tables):
        row = t.iloc[0]
        x = t["wavelength_nm"].to_numpy(dtype=float)
        y = t["irradiance_mean"].to_numpy(dtype=float)
        s = t["irradiance_std"].to_numpy(dtype=float)
        y = normalize_curve(y)
        if NORMALIZE_Y:
            base = float(np.nanmax(t["irradiance_mean"].to_numpy(dtype=float)))
            if np.isfinite(base) and base > 0:
                s = s / base
            else:
                s = np.zeros_like(s, dtype=float)
        x_all.append(x)
        ax.plot(x, y, label=grouped_label(row), color=color, alpha=0.95)
        if SHOW_STD_BAND:
            n = t["n_curves"].to_numpy()
            mask = np.isfinite(s) & (n >= 2)
            if mask.any():
                ax.fill_between(x[mask], (y - s)[mask], (y + s)[mask], color=color, alpha=0.14)

    x_full = np.concatenate(x_all) if x_all else np.array([], dtype=float)
    ax.set_title(title)
    ax.set_xlabel(f"Wavelength {spectral_interval_label(x_full)}")
    ax.set_ylabel("Normalized Irradiance (a.u.)" if NORMALIZE_Y else "Irradiance (W m$^{-2}$ nm$^{-1}$)")
    style_axes(ax, grid_axis="both")
    if NORMALIZE_Y:
        ax.set_ylim(0, 1.05)
    if tables:
        ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def write_individual_figures(df: pd.DataFrame) -> List[Path]:
    written: List[Path] = []
    for dataset, ds in df.groupby("dataset", dropna=False):
        dataset_name = str(dataset)
        out_dir = spectral_individual_dir(dataset_name)
        clear_pngs(out_dir)
        for index, (sample_id, g) in enumerate(
            ds.groupby("sample_id", dropna=False),
            start=1,
        ):
            g = g.sort_values("wavelength_nm")
            row = g.iloc[0]
            x = g["wavelength_nm"].to_numpy(dtype=float)
            y = normalize_curve(g["irradiance_W_m2_nm"].to_numpy(dtype=float))
            fig, ax = plt.subplots(figsize=(10, 5.4))
            ax.plot(x, y, color=get_palette(1)[0], alpha=0.95)
            ax.set_title(
                " | ".join(
                    [
                        str(row.get("dataset", "")),
                        str(row.get("param_set", "")),
                        str(row.get("channel", "")),
                        str(sample_id),
                    ]
                )
            )
            ax.set_xlabel(f"Wavelength {spectral_interval_label(x)}")
            ax.set_ylabel("Normalized Irradiance (a.u.)" if NORMALIZE_Y else "Irradiance (W m$^{-2}$ nm$^{-1}$)")
            style_axes(ax, grid_axis="both")
            if NORMALIZE_Y:
                ax.set_ylim(0, 1.05)
            out_path = out_dir / f"Fig{index}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=DPI)
            plt.close(fig)
            written.append(out_path)
    return written


def write_composed_figures(df: pd.DataFrame) -> List[Path]:
    written: List[Path] = []
    group_keys = ["dataset", "param_set"]
    grouped = df.groupby(group_keys, dropna=False)
    by_dataset: Dict[str, List[tuple[object, object, pd.DataFrame]]] = {}
    for (dataset, param_set), g in grouped:
        key = str(dataset)
        by_dataset.setdefault(key, []).append((dataset, param_set, g.copy()))

    for dataset_name, blocks in by_dataset.items():
        out_dir = spectral_composed_dir(dataset_name)
        clear_pngs(out_dir)
        for index, (_, param_set, g) in enumerate(blocks, start=1):
            curves = list(g.groupby("sample_id", dropna=False))
            palette = get_palette(len(curves))
            fig, ax = plt.subplots(figsize=(10, 5.4))
            x_all: List[np.ndarray] = []
            for color, (_, c) in zip(palette, curves):
                c = c.sort_values("wavelength_nm")
                row = c.iloc[0]
                x = c["wavelength_nm"].to_numpy(dtype=float)
                y = normalize_curve(c["irradiance_W_m2_nm"].to_numpy(dtype=float))
                x_all.append(x)
                label_parts: List[str] = []
                channel = str(row.get("channel", "")).strip()
                if channel and channel.lower() not in {"bulk", "irradiance"}:
                    label_parts.append(channel)
                trial = row.get("trial")
                if pd.notna(trial):
                    try:
                        label_parts.append(f"trial {int(trial)}")
                    except Exception:
                        label_parts.append(f"trial {trial}")
                if not label_parts:
                    label_parts.append(str(row.get("sample_id", "sample")))
                ax.plot(x, y, label=" | ".join(label_parts), color=color, alpha=0.95)

            x_full = np.concatenate(x_all) if x_all else np.array([], dtype=float)
            ax.set_title(f"{dataset_name} | {param_set}")
            ax.set_xlabel(f"Wavelength {spectral_interval_label(x_full)}")
            ax.set_ylabel("Normalized Irradiance (a.u.)" if NORMALIZE_Y else "Irradiance (W m$^{-2}$ nm$^{-1}$)")
            style_axes(ax, grid_axis="both")
            if NORMALIZE_Y:
                ax.set_ylim(0, 1.05)
            if len(curves) > 1:
                ax.legend(loc="best", fontsize=9, frameon=True)

            out_path = out_dir / f"Fig{index}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=DPI)
            plt.close(fig)
            written.append(out_path)
    return written


def write_compared_figures(df: pd.DataFrame) -> List[Path]:
    written: List[Path] = []
    avg_tables: List[pd.DataFrame] = []
    for (dataset, param_set, channel), g in df.groupby(["dataset", "param_set", "channel"], dropna=False):
        avg_tables.append(average_group(str(dataset), str(param_set), str(channel), g))

    clear_pngs(spectral_compared_dir("meta"))
    clear_pngs(spectral_compared_dir("air"))
    clear_pngs(spectral_compared_dir("diameter"))

    if avg_tables:
        meta_path = spectral_compared_dir("meta") / "Fig1.png"
        plot_tables(avg_tables, meta_path, "Averaged Spectra by Dataset / Parameter Set / Channel")
        written.append(meta_path)

    for dataset_name in sorted(df["dataset"].astype(str).unique()):
        ds_tables = [t for t in avg_tables if str(t.iloc[0]["dataset"]) == dataset_name]
        if not ds_tables:
            continue
        ds_path = spectral_compared_dir(dataset_name) / "Fig1.png"
        plot_tables(ds_tables, ds_path, f"Averaged Spectra - {dataset_name}")
        written.append(ds_path)
    return written


def main() -> int:
    apply_publication_style()
    ensure_all_scope_layouts()

    if not INPUT_LONG.exists():
        print(f"Missing {INPUT_LONG}. Run preprocess.py first.")
        return 1

    df = pd.read_csv(INPUT_LONG)
    required = {"dataset", "sample_id", "param_set", "channel", "wavelength_nm", "irradiance_W_m2_nm"}
    if not required.issubset(df.columns):
        print(f"{INPUT_LONG} must have columns: {sorted(required)}")
        return 2

    written: List[Path] = []
    written.extend(write_individual_figures(df))
    written.extend(write_composed_figures(df))
    written.extend(write_compared_figures(df))

    print("Wrote spectral chart outputs:")
    for path in written:
        print(f"  {path}")
    print(f"Done. figures={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
