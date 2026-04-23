#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.output_paths import SCOPES, ensure_all_scope_layouts, metadata_section_dir, spectral_diagnostics_dir
from plots.style import apply_publication_style, get_palette, spectral_interval_label, style_axes, to_species_label

RAW_DIR = metadata_section_dir("meta", "spectral")
AVERAGED_CURVES_CSV = RAW_DIR / "averaged_curves_long.csv"
AVERAGED_PEAKS_CSV = RAW_DIR / "averaged_peaks_top10.csv"
TRIAL_PEAKS_CSV = RAW_DIR / "trial_peaks_top10.csv"
NIST_MATCHES_CSV = RAW_DIR / "nist_matches_top3.csv"
TARGET_MATCHES_CSV = RAW_DIR / "target_species_peak_matches.csv"
OUT_DIR = spectral_diagnostics_dir("meta")
DPI = 220


def group_label(df: pd.DataFrame) -> pd.Series:
    return (
        df["dataset"].astype(str).str.strip()
        + " | "
        + df["param_set"].astype(str).str.strip()
        + " | "
        + df["channel"].astype(str).str.strip()
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def fig_path(index: int) -> Path:
    return OUT_DIR / f"Fig{index}.png"


def write_note_figure(path: Path, message: str, size: tuple[float, float] = (8, 4)) -> Path:
    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_averaged_spectra(curves: pd.DataFrame) -> Path:
    grouped = list(curves.groupby(["dataset", "param_set", "channel"], dropna=False))
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = get_palette(len(grouped))
    x_all: List[np.ndarray] = []

    for color, ((dataset, param_set, channel), g) in zip(palette, grouped):
        g = g.sort_values("wavelength_nm")
        x = g["wavelength_nm"].to_numpy(dtype=float)
        x_all.append(x)
        label = f"{dataset}:{param_set}:{channel}"
        ax.plot(x, g["irradiance_mean"].to_numpy(dtype=float), color=color, alpha=0.92, label=label)

    x_full = np.concatenate(x_all) if x_all else np.array([], dtype=float)
    ax.set_title("Averaged Spectra Overview")
    ax.set_xlabel(f"Wavelength {spectral_interval_label(x_full)}")
    ax.set_ylabel("Mean Irradiance (W m$^{-2}$ nm$^{-1}$)")
    style_axes(ax, grid_axis="both")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    out_path = fig_path(1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_peak_map(peaks: pd.DataFrame) -> Path:
    out_path = fig_path(2)
    if peaks.empty:
        return write_note_figure(out_path, "No averaged peaks found.", (10, 4))

    data = peaks.copy()
    data["group"] = group_label(data)
    groups = sorted(data["group"].unique().tolist())
    y_map = {g: i for i, g in enumerate(groups)}
    data["y"] = data["group"].map(y_map).astype(int)

    pmin = float(data["peak_intensity"].min())
    pmax = float(data["peak_intensity"].max())
    if pmax > pmin:
        size = 40 + 220 * ((data["peak_intensity"] - pmin) / (pmax - pmin))
    else:
        size = np.full(len(data), 110.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sc = ax.scatter(
        data["peak_wavelength_nm_0p1"],
        data["y"],
        s=size,
        c=data["peak_rank"],
        cmap="viridis_r",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.35,
    )
    ax.set_yticks(list(range(len(groups))))
    ax.set_yticklabels(groups)
    ax.set_xlabel(f"Peak Wavelength {spectral_interval_label(data['peak_wavelength_nm_0p1'].to_numpy(dtype=float))}")
    ax.set_ylabel("Dataset | Param Set | Channel")
    ax.set_title("Top-10 Averaged Peaks by Group")
    style_axes(ax, grid_axis="x")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Peak Rank (1 = strongest)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_trial_repeatability(trial_peaks: pd.DataFrame) -> Path:
    out_path = fig_path(3)
    if trial_peaks.empty:
        return write_note_figure(out_path, "No trial peaks found.", (9, 4))

    data = trial_peaks.copy()
    data["group"] = group_label(data)
    spread = (
        data.groupby(["group", "peak_rank"], dropna=False)["peak_wavelength_nm_0p1"]
        .agg(["count", "std"])
        .reset_index()
        .rename(columns={"count": "n_trials", "std": "wavelength_std_nm"})
    )
    spread = spread[spread["n_trials"] >= 2].copy()
    if spread.empty:
        return write_note_figure(out_path, "No groups with >=2 trials for repeatability.", (10, 4))

    summary = (
        spread.groupby("group", dropna=False)["wavelength_std_nm"]
        .median()
        .reset_index()
        .sort_values("wavelength_std_nm", ascending=True, ignore_index=True)
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = get_palette(len(summary))
    ax.barh(summary["group"], summary["wavelength_std_nm"], color=palette, alpha=0.9)
    ax.set_xlabel("Median Trial Peak Spread (nm, std across trials)")
    ax.set_ylabel("Dataset | Param Set | Channel")
    ax.set_title("Trial-to-Trial Peak Wavelength Repeatability")
    style_axes(ax, grid_axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_nist_coverage(averaged_peaks: pd.DataFrame, nist_matches: pd.DataFrame, target_matches: pd.DataFrame) -> Path:
    out_path = fig_path(4)
    if not target_matches.empty and {"dataset", "param_set", "channel", "matched"}.issubset(target_matches.columns):
        d = target_matches.copy()
        d["group"] = group_label(d)
        totals = d.groupby("group", dropna=False).size().rename("total_targets").reset_index()
        matched = (
            d[d["matched"].astype(bool)]
            .groupby("group", dropna=False)
            .size()
            .rename("matched_targets")
            .reset_index()
        )
        cov = totals.merge(matched, on="group", how="left")
        cov["matched_targets"] = pd.to_numeric(cov["matched_targets"], errors="coerce").fillna(0).astype(int)
        cov["unmatched_targets"] = cov["total_targets"] - cov["matched_targets"]
        cov = cov.sort_values("matched_targets", ascending=False, ignore_index=True)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        y = np.arange(len(cov))
        ax.barh(y, cov["matched_targets"], color="#2a9d8f", label="Matched")
        ax.barh(y, cov["unmatched_targets"], left=cov["matched_targets"], color="#bfc0c0", label="Unmatched")
        ax.set_yticks(y)
        ax.set_yticklabels(cov["group"])
        ax.set_xlabel("Count of Target Species Lines")
        ax.set_ylabel("Dataset | Param Set | Channel")
        ax.set_title("Target-Species Match Coverage by Group")
        ax.legend(loc="lower right")
        style_axes(ax, grid_axis="x")
        fig.tight_layout()
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        return out_path

    if averaged_peaks.empty:
        return write_note_figure(out_path, "No averaged peaks found.", (8, 4))

    base = averaged_peaks.copy()
    base["group"] = group_label(base)
    totals = base.groupby("group", dropna=False).size().rename("total_peaks").reset_index()

    if nist_matches.empty:
        matched = totals[["group"]].copy()
        matched["matched_peaks"] = 0
    else:
        m = nist_matches.copy()
        m["group"] = group_label(m)
        matched = (
            m[["group", "peak_rank", "peak_wavelength_nm_0p1"]]
            .drop_duplicates()
            .groupby("group", dropna=False)
            .size()
            .rename("matched_peaks")
            .reset_index()
        )

    cov = totals.merge(matched, on="group", how="left")
    cov["matched_peaks"] = pd.to_numeric(cov["matched_peaks"], errors="coerce").fillna(0).astype(int)
    cov["unmatched_peaks"] = cov["total_peaks"] - cov["matched_peaks"]
    cov = cov.sort_values("matched_peaks", ascending=False, ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(cov))
    ax.barh(y, cov["matched_peaks"], color="#2a9d8f", label="Matched")
    ax.barh(y, cov["unmatched_peaks"], left=cov["matched_peaks"], color="#bfc0c0", label="Unmatched")
    ax.set_yticks(y)
    ax.set_yticklabels(cov["group"])
    ax.set_xlabel("Count of Top-10 Averaged Peaks")
    ax.set_ylabel("Dataset | Param Set | Channel")
    ax.set_title("NIST Match Coverage by Group")
    ax.legend(loc="lower right")
    style_axes(ax, grid_axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_nist_top1_species(nist_matches: pd.DataFrame, target_matches: pd.DataFrame) -> Path:
    out_path = fig_path(5)
    if not target_matches.empty and "species" in target_matches.columns:
        d = target_matches.copy()
        if "matched" in d.columns:
            d = d[d["matched"].astype(bool)]
        counts = d["species"].astype(str).value_counts().head(12)
        if counts.empty:
            return write_note_figure(out_path, "No target species matches found.", (8, 4))

        fig, ax = plt.subplots(figsize=(10, 5))
        palette = get_palette(len(counts))
        labels = [to_species_label(v) for v in counts.index.tolist()]
        ax.bar(labels, counts.to_numpy(dtype=float), color=palette, alpha=0.9)
        ax.set_xlabel("Target Species")
        ax.set_ylabel("Count of Matched Target Lines")
        ax.set_title("Most Frequent Target-Species Matches")
        ax.tick_params(axis="x", labelrotation=35, labelsize=10.8)
        style_axes(ax, grid_axis="y")

        fig.tight_layout()
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        return out_path

    if nist_matches.empty or "candidate_rank" not in nist_matches.columns:
        return write_note_figure(out_path, "No NIST matches found.", (8, 4))

    label_col = "nist_species" if "nist_species" in nist_matches.columns else None
    if label_col is None and "nist_spectra_query" in nist_matches.columns:
        label_col = "nist_spectra_query"
    if label_col is None and "nist_element" in nist_matches.columns:
        label_col = "nist_element"
    if label_col is None:
        label_col = nist_matches.columns[0]

    top = nist_matches[nist_matches["candidate_rank"] == 1].copy()
    top["species_label"] = top[label_col].fillna("unknown").astype(str)
    counts = top["species_label"].value_counts().head(12)
    if counts.empty:
        return write_note_figure(out_path, "No top-1 NIST candidates found.", (8, 4))

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = get_palette(len(counts))
    labels = [to_species_label(v) for v in counts.index.tolist()]
    ax.bar(labels, counts.to_numpy(dtype=float), color=palette, alpha=0.9)
    ax.set_xlabel("Top-1 NIST Candidate Species")
    ax.set_ylabel("Count of Averaged Peaks")
    ax.set_title("Most Frequent Top-1 NIST Candidates")
    ax.tick_params(axis="x", labelrotation=35, labelsize=10.8)
    style_axes(ax, grid_axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def main() -> int:
    apply_publication_style()
    ensure_all_scope_layouts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()

    averaged_curves = load_csv(AVERAGED_CURVES_CSV)
    averaged_peaks = load_csv(AVERAGED_PEAKS_CSV)
    trial_peaks = load_csv(TRIAL_PEAKS_CSV)
    nist_matches = load_csv(NIST_MATCHES_CSV) if NIST_MATCHES_CSV.exists() else pd.DataFrame()
    target_matches = load_csv(TARGET_MATCHES_CSV) if TARGET_MATCHES_CSV.exists() else pd.DataFrame()

    figure_paths: List[Path] = [
        plot_averaged_spectra(averaged_curves),
        plot_peak_map(averaged_peaks),
        plot_trial_repeatability(trial_peaks),
        plot_nist_coverage(averaged_peaks, nist_matches, target_matches),
        plot_nist_top1_species(nist_matches, target_matches),
    ]

    print("Wrote diagnostic figures:")
    for p in figure_paths:
        print(f"  {p}")

    from plots.labeled_spectra import write_labeled_assets

    for scope in SCOPES:
        write_labeled_assets(scope)
    print("Wrote labeled spectral assets under output/*/spectral/labels with metadata traceability CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
