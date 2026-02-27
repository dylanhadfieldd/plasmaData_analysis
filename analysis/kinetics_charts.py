#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.plot_style import apply_publication_style, get_palette, style_axes

IN_CSV = Path("output/kinetics_features.csv")
IN_MODEL_PRED = Path("stats/models/target_regression_predictions.csv")
OUT_DIR = Path("output/kinetics_charts")
META_DIR = OUT_DIR / "meta"
DPI = 200
SAFE_TEXT_RE = re.compile(r"[^a-z0-9]+")

INDEX_COLS = ["sterilization_index", "healing_index", "healing_sterilization_ratio"]
INDEX_LABELS = {
    "sterilization_index": "Sterilization Proxy Index",
    "healing_index": "Healing Proxy Index",
    "healing_sterilization_ratio": "Healing/Sterilization Ratio",
}

SPECIES_LABELS = {
    "oh_309": r"OH (309 nm)",
    "no_gamma": r"NO-$\gamma$ (220-250 nm)",
    "n2_337": r"N$_2$ (337 nm)",
    "n2plus_391": r"N$_2^+$ (391 nm)",
    "uvc_continuum": "UVC Continuum (200-280 nm)",
    "hgamma_434": r"H$\gamma$ (434 nm)",
    "hbeta_486": r"H$\beta$ (486 nm)",
    "n2_sps": r"N$_2$ SPS",
    "balmer": "H Balmer",
}

FEATURE_LABELS = {
    "oh_to_n2_337_ratio": r"OH / N$_2$(337) Ratio",
    "n2_337_to_n2plus_391_ratio": r"N$_2$(337) / N$_2^+$(391) Ratio",
    "n2_sps_to_n2plus_ratio": r"N$_2$ SPS / N$_2^+$ Ratio",
    "no_gamma_to_oh_ratio": r"NO-$\gamma$ / OH Ratio",
    "uvc_to_n2_337_ratio": r"UVC / N$_2$(337) Ratio",
    "balmer_to_n2plus_ratio": r"H Balmer / N$_2^+$ Ratio",
    "target": "Observed Target",
    "pred_loo": "Predicted Target (LOO)",
    "residual_loo": "Residual (Target - LOO Prediction)",
    "delta_sterilization_vs_baseline": "Sterilization Proxy Delta vs Baseline",
    "delta_healing_vs_baseline": "Healing Proxy Delta vs Baseline",
}


def safe_name(text: str) -> str:
    out = SAFE_TEXT_RE.sub("_", str(text).strip().lower()).strip("_")
    return out or "plot"


def readable_species(col: str) -> str:
    token = col.replace("dose_", "").strip().lower()
    return SPECIES_LABELS.get(token, token.replace("_", " ").title())


def metric_label(col: str) -> str:
    if col in INDEX_LABELS:
        return INDEX_LABELS[col]
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    if col.startswith("dose_"):
        return f"{readable_species(col)} Dose"
    if col.startswith("gen_norm_"):
        token = col.replace("gen_norm_", "").strip().lower()
        base = SPECIES_LABELS.get(token, token.replace("_", " ").title())
        return f"{base} Generation (normalized)"
    if col.startswith("c_ss_"):
        token = col.replace("c_ss_", "").strip().lower()
        base = SPECIES_LABELS.get(token, token.replace("_", " ").title())
        return f"{base} Steady-State Proxy"
    return col.replace("_", " ").title()


def ordered_unique(values: Sequence[object]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def color_map_for_values(values: Sequence[str], cmap_name: str = "tab10") -> Dict[str, object]:
    unique = ordered_unique(values)
    palette = get_palette(len(unique), cmap_name)
    return {v: palette[i] for i, v in enumerate(unique)}


def marker_map_for_values(values: Sequence[str]) -> Dict[str, str]:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
    unique = ordered_unique(values)
    return {v: markers[i % len(markers)] for i, v in enumerate(unique)}


def plot_dataset_index_trends(dataset: str, df: pd.DataFrame, out_path: Path) -> None:
    by_param = (
        df.groupby("param_set", dropna=False)[INDEX_COLS]
        .agg(["mean", "std"])
        .sort_index()
    )
    params = by_param.index.astype(str).tolist()
    x = np.arange(len(params), dtype=float)

    fig, axes = plt.subplots(1, len(INDEX_COLS), figsize=(15, 4.5))
    if len(INDEX_COLS) == 1:
        axes = [axes]

    metric_colors = get_palette(len(INDEX_COLS))
    for ax, metric, color in zip(axes, INDEX_COLS, metric_colors):
        means = by_param[(metric, "mean")].to_numpy(dtype=float)
        stds = by_param[(metric, "std")].fillna(0.0).to_numpy(dtype=float)

        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            linestyle="-",
            capsize=3,
            color=color,
            ecolor=color,
        )

        # Overlay raw sample points with slight jitter for replicate visibility.
        for i, param in enumerate(params):
            vals = pd.to_numeric(df.loc[df["param_set"].astype(str) == param, metric], errors="coerce").dropna()
            if vals.empty:
                continue
            jitter = np.linspace(-0.06, 0.06, len(vals))
            ax.scatter(
                np.full(len(vals), i, dtype=float) + jitter,
                vals.to_numpy(dtype=float),
                s=20,
                alpha=0.85,
                color=color,
                edgecolors="white",
                linewidth=0.4,
            )

        ax.set_title(metric_label(metric))
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=30, ha="right")
        style_axes(ax, grid_axis="both")

    fig.suptitle(f"Kinetics Indices by Parameter Set ({dataset})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_dataset_channel_heatmaps(dataset: str, df: pd.DataFrame, out_path: Path) -> None:
    params = sorted(df["param_set"].astype(str).unique().tolist())
    channels = sorted(df["channel"].astype(str).unique().tolist())

    fig, axes = plt.subplots(1, len(INDEX_COLS), figsize=(15, 5))
    if len(INDEX_COLS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, INDEX_COLS):
        mat = (
            df.groupby(["param_set", "channel"], dropna=False)[metric]
            .mean()
            .unstack("channel")
            .reindex(index=params, columns=channels)
        )
        arr = mat.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu")
        ax.set_title(metric_label(metric))
        ax.set_yticks(np.arange(len(params)))
        ax.set_yticklabels(params)
        ax.set_xticks(np.arange(len(channels)))
        ax.set_xticklabels(channels, rotation=35, ha="right")

        vmax_raw = np.nanmax(arr) if arr.size else np.nan
        vmax = float(vmax_raw) if np.isfinite(vmax_raw) else 1.0
        threshold = 0.55 * vmax if np.isfinite(vmax) and vmax > 0 else 0.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    text_color = "white" if arr[i, j] >= threshold else "#1c1c1c"
                    ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8, color=text_color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Kinetics Index Heatmaps by Param/Channel ({dataset})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_dataset_dose_composition(dataset: str, df: pd.DataFrame, dose_cols: List[str], out_path: Path) -> None:
    if not dose_cols:
        return

    by_param = df.groupby("param_set", dropna=False)[dose_cols].mean().sort_index()
    params = by_param.index.astype(str).tolist()

    dose_values = by_param.to_numpy(dtype=float)
    totals = np.nansum(dose_values, axis=1)
    totals_safe = np.where(totals > 0, totals, np.nan)
    frac = dose_values / totals_safe[:, None]
    frac = np.nan_to_num(frac, nan=0.0)

    colors = get_palette(len(dose_cols), "tab20")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(params))

    bottom = np.zeros(len(params), dtype=float)
    for i, col in enumerate(dose_cols):
        vals = frac[:, i]
        ax1.bar(x, vals, bottom=bottom, width=0.75, label=metric_label(col), color=colors[i])
        bottom += vals

    ax1.set_title("Dose Composition (Fraction of Total)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(params, rotation=30, ha="right")
    ax1.set_ylim(0, 1.02)
    style_axes(ax1, grid_axis="y")
    ax1.legend(loc="upper left", fontsize=8, ncol=2, frameon=True)

    ax2.bar(x, totals, width=0.75, color="steelblue")
    ax2.set_title("Total Dose Proxy (Sum of Species Doses)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(params, rotation=30, ha="right")
    ax2.set_ylabel("Arbitrary dose units")
    style_axes(ax2, grid_axis="y")

    fig.suptitle(f"Species Dose Composition by Parameter Set ({dataset})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_dataset_phase_space(dataset: str, df: pd.DataFrame, out_path: Path) -> None:
    param_labels = df["param_set"].astype(str).tolist()
    channel_labels = df["channel"].astype(str).tolist()
    color_map = color_map_for_values(param_labels, "tab10")
    marker_map = marker_map_for_values(channel_labels)

    fig, ax = plt.subplots(figsize=(7.2, 6))
    for _, row in df.iterrows():
        param = str(row["param_set"])
        channel = str(row["channel"])
        x = float(row["sterilization_index"])
        y = float(row["healing_index"])
        marker = marker_map[channel]
        ax.scatter(x, y, s=52, color=color_map[param], marker=marker, alpha=0.9, edgecolors="white", linewidth=0.45)

    # Draw y=x for quick balance checks.
    vals_x = pd.to_numeric(df["sterilization_index"], errors="coerce")
    vals_y = pd.to_numeric(df["healing_index"], errors="coerce")
    lo = float(np.nanmin([vals_x.min(), vals_y.min()]))
    hi = float(np.nanmax([vals_x.max(), vals_y.max()]))
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="black", alpha=0.7)

    ax.set_xlabel("Sterilization Index")
    ax.set_ylabel("Healing Index")
    ax.set_title(f"Kinetics Phase Space ({dataset})")
    style_axes(ax, grid_axis="both")

    param_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[p], label=f"param: {p}", markersize=7)
        for p in ordered_unique(param_labels)
    ]
    channel_handles = [
        plt.Line2D([0], [0], marker=marker_map[ch], linestyle="", color="gray", label=f"channel: {ch}", markersize=7)
        for ch in ordered_unique(channel_labels)
    ]
    if param_handles:
        leg1 = ax.legend(handles=param_handles, loc="upper left", fontsize=8, frameon=True)
        ax.add_artist(leg1)
    if channel_handles:
        ax.legend(handles=channel_handles, loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def pareto_frontier(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    # Maximize both axes.
    order = np.argsort(points[:, 0])  # sort by x ascending
    sorted_points = points[order]
    frontier: List[List[float]] = []
    best_y = -np.inf
    for x, y in sorted_points:
        if y >= best_y:
            frontier.append([float(x), float(y)])
            best_y = y
    return np.asarray(frontier, dtype=float)


def plot_meta_phase_space(df: pd.DataFrame, out_path: Path) -> None:
    dataset_labels = df["dataset"].astype(str).tolist()
    param_labels = df["param_set"].astype(str).tolist()
    color_map = color_map_for_values(dataset_labels, "Set2")
    marker_map = marker_map_for_values(param_labels)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for _, row in df.iterrows():
        ds = str(row["dataset"])
        param = str(row["param_set"])
        ax.scatter(
            float(row["sterilization_index"]),
            float(row["healing_index"]),
            color=color_map[ds],
            marker=marker_map[param],
            s=52,
            alpha=0.9,
            edgecolors="white",
            linewidth=0.45,
        )

    pts = df[["sterilization_index", "healing_index"]].to_numpy(dtype=float)
    frontier = pareto_frontier(pts)
    if len(frontier) > 1:
        ax.plot(frontier[:, 0], frontier[:, 1], color="black", linewidth=1.2, linestyle="--", label="Pareto frontier")

    ax.set_xlabel("Sterilization Index")
    ax.set_ylabel("Healing Index")
    ax.set_title("Meta Kinetics Phase Space (All Datasets)")
    style_axes(ax, grid_axis="both")

    ds_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[d], label=f"dataset: {d}", markersize=7)
        for d in ordered_unique(dataset_labels)
    ]
    param_handles = [
        plt.Line2D([0], [0], marker=marker_map[p], linestyle="", color="gray", label=f"param: {p}", markersize=7)
        for p in ordered_unique(param_labels)
    ]
    if ds_handles:
        leg1 = ax.legend(handles=ds_handles, loc="upper left", fontsize=8, frameon=True)
        ax.add_artist(leg1)
    if param_handles:
        ax.legend(handles=param_handles, loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_meta_delta_bars(df: pd.DataFrame, out_path: Path, out_csv: Path) -> None:
    summary = (
        df.groupby(["dataset", "param_set"], dropna=False)[["sterilization_index", "healing_index"]]
        .mean()
        .reset_index()
    )

    rows: List[Dict[str, object]] = []
    for dataset, g in summary.groupby("dataset", dropna=False):
        g = g.sort_values("param_set").reset_index(drop=True)
        if g.empty:
            continue
        baseline = g.iloc[0]
        b_ster = float(baseline["sterilization_index"])
        b_heal = float(baseline["healing_index"])
        b_param = str(baseline["param_set"])

        for _, row in g.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "baseline_param": b_param,
                    "param_set": row["param_set"],
                    "delta_sterilization_vs_baseline": float(row["sterilization_index"]) - b_ster,
                    "delta_healing_vs_baseline": float(row["healing_index"]) - b_heal,
                }
            )

    delta_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    delta_df.to_csv(out_csv, index=False)
    if delta_df.empty:
        return

    labels = [f"{r.dataset}:{r.param_set}" for r in delta_df.itertuples(index=False)]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.85), 5))
    ax.bar(
        x - width / 2,
        delta_df["delta_sterilization_vs_baseline"],
        width=width,
        label=metric_label("delta_sterilization_vs_baseline"),
    )
    ax.bar(
        x + width / 2,
        delta_df["delta_healing_vs_baseline"],
        width=width,
        label=metric_label("delta_healing_vs_baseline"),
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Proxy Delta vs Dataset Baseline")
    ax.set_title("Meta Delta Analysis vs Baseline Parameter Set")
    style_axes(ax, grid_axis="y")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, cols: List[str], out_path: Path, out_csv: Path, title: str) -> None:
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return

    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_csv)

    fig, ax = plt.subplots(figsize=(max(7, len(cols) * 0.8), max(6, len(cols) * 0.65)))
    im = ax.imshow(corr.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([metric_label(c) for c in cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels([metric_label(c) for c in cols])
    ax.set_title(title)

    arr = corr.to_numpy(dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_model_diagnostics(pred_path: Path, out_path: Path) -> int:
    if not pred_path.exists():
        print(f"Skipping model diagnostics (missing {pred_path})")
        return 0

    pred = pd.read_csv(pred_path)
    required = {"target", "pred_loo", "residual_loo", "param_set"}
    if not required.issubset(pred.columns):
        print(f"Skipping model diagnostics ({pred_path} missing required columns)")
        return 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.scatter(pred["target"], pred["pred_loo"], s=32, color=get_palette(1)[0], alpha=0.88, edgecolors="white", linewidth=0.45)
    lo = float(min(pred["target"].min(), pred["pred_loo"].min()))
    hi = float(max(pred["target"].max(), pred["pred_loo"].max()))
    ax1.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="black")
    ax1.set_xlabel("Observed target")
    ax1.set_ylabel("Predicted target (LOO)")
    ax1.set_title("LOO Calibration")
    style_axes(ax1, grid_axis="both")

    params = sorted(pred["param_set"].astype(str).unique().tolist())
    groups = [pred.loc[pred["param_set"].astype(str) == p, "residual_loo"].to_numpy(dtype=float) for p in params]
    bplot = ax2.boxplot(groups, labels=params, showfliers=True, patch_artist=True)
    box_colors = get_palette(len(bplot["boxes"]))
    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax2.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax2.set_ylabel("Residual (target - pred_loo)")
    ax2.set_title("Residuals by Parameter Set")
    ax2.tick_params(axis="x", rotation=30)
    style_axes(ax2, grid_axis="y")

    fig.suptitle("Kinetics Model Diagnostics")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return 0


def main() -> int:
    apply_publication_style()

    if not IN_CSV.exists():
        print(f"Missing {IN_CSV}. Run kinetics.py first.")
        return 1

    df = pd.read_csv(IN_CSV)
    if df.empty:
        print(f"Input CSV is empty: {IN_CSV}")
        return 2

    required = {"dataset", "param_set", "channel"} | set(INDEX_COLS)
    if not required.issubset(df.columns):
        print(f"{IN_CSV} missing required columns: {sorted(required)}")
        return 2

    dose_cols = [c for c in df.columns if c.startswith("dose_")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # Save reusable summaries for reporting.
    summary_param_channel = (
        df.groupby(["dataset", "param_set", "channel"], dropna=False)[INDEX_COLS + dose_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_param_channel.columns = ["__".join([str(x) for x in col if str(x) != ""]).strip("__") for col in summary_param_channel.columns]
    summary_param_channel.to_csv(OUT_DIR / "kinetics_param_channel_summary.csv", index=False)

    summary_param = (
        df.groupby(["dataset", "param_set"], dropna=False)[INDEX_COLS + dose_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_param.columns = ["__".join([str(x) for x in col if str(x) != ""]).strip("__") for col in summary_param.columns]
    summary_param.to_csv(OUT_DIR / "kinetics_param_summary.csv", index=False)

    for dataset, g in df.groupby("dataset", dropna=False):
        ds = str(dataset)
        ds_dir = OUT_DIR / safe_name(ds)
        ds_dir.mkdir(parents=True, exist_ok=True)

        plot_dataset_index_trends(ds, g, ds_dir / "indices_by_param.png")
        plot_dataset_channel_heatmaps(ds, g, ds_dir / "indices_param_channel_heatmaps.png")
        plot_dataset_dose_composition(ds, g, dose_cols, ds_dir / "dose_composition.png")
        plot_dataset_phase_space(ds, g, ds_dir / "phase_space.png")

    plot_meta_phase_space(df, META_DIR / "meta_phase_space.png")
    plot_meta_delta_bars(
        df,
        META_DIR / "meta_delta_vs_baseline.png",
        META_DIR / "meta_delta_vs_baseline.csv",
    )
    plot_correlation_heatmap(
        df,
        dose_cols + INDEX_COLS,
        META_DIR / "meta_correlation_heatmap.png",
        META_DIR / "meta_correlation_matrix.csv",
        "Meta Correlation: Doses and Kinetics Indices",
    )
    plot_model_diagnostics(IN_MODEL_PRED, META_DIR / "meta_model_diagnostics.png")

    # Correlation heatmap including target if model predictions are available.
    if IN_MODEL_PRED.exists():
        pred = pd.read_csv(IN_MODEL_PRED)
        merge_cols = ["sample_id", "target", "pred_loo", "residual_loo"]
        merge_cols = [c for c in merge_cols if c in pred.columns]
        if "sample_id" in merge_cols:
            merged = df.merge(pred[merge_cols], on="sample_id", how="left")
            if "target" in merged.columns:
                plot_correlation_heatmap(
                    merged,
                    dose_cols + INDEX_COLS + [c for c in ["target", "pred_loo", "residual_loo"] if c in merged.columns],
                    META_DIR / "meta_correlation_with_target.png",
                    META_DIR / "meta_correlation_with_target.csv",
                    "Meta Correlation Including Target and Model Outputs",
                )

    print(f"Wrote {OUT_DIR / 'kinetics_param_channel_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'kinetics_param_summary.csv'}")
    for dataset in sorted(df["dataset"].astype(str).unique()):
        ds_dir = OUT_DIR / safe_name(dataset)
        print(f"Wrote {ds_dir / 'indices_by_param.png'}")
        print(f"Wrote {ds_dir / 'indices_param_channel_heatmaps.png'}")
        print(f"Wrote {ds_dir / 'dose_composition.png'}")
        print(f"Wrote {ds_dir / 'phase_space.png'}")
    print(f"Wrote {META_DIR / 'meta_phase_space.png'}")
    print(f"Wrote {META_DIR / 'meta_delta_vs_baseline.png'}")
    print(f"Wrote {META_DIR / 'meta_delta_vs_baseline.csv'}")
    print(f"Wrote {META_DIR / 'meta_correlation_heatmap.png'}")
    print(f"Wrote {META_DIR / 'meta_correlation_matrix.csv'}")
    print(f"Wrote {META_DIR / 'meta_model_diagnostics.png'}")
    if (META_DIR / "meta_correlation_with_target.png").exists():
        print(f"Wrote {META_DIR / 'meta_correlation_with_target.png'}")
        print(f"Wrote {META_DIR / 'meta_correlation_with_target.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
