#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from analysis.plot_style import apply_publication_style, get_palette, style_axes

IN_CSV = Path("output/features.csv")
OUTPUT_ROOT = Path("output")
N_COMPONENTS = 5
SCALE = False
INCLUDE_COLS: Optional[List[str]] = None
EXCLUDE_NUMERIC_COLS = {"trial"}
DIAMETER_DATASET = "diameter"
TIP_MIDDLE_BASE_CHANNELS = {"tip", "middle", "base"}


def select_variable_columns(df: pd.DataFrame) -> List[str]:
    if INCLUDE_COLS:
        return [c for c in INCLUDE_COLS if c in df.columns]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in EXCLUDE_NUMERIC_COLS]


def impute_nan_matrix(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, np.nan)
    col_means = np.zeros(x.shape[1], dtype=float)
    valid_cols = np.any(np.isfinite(x), axis=0)
    if np.any(valid_cols):
        col_means[valid_cols] = np.nanmean(x[:, valid_cols], axis=0)
    mask = np.isnan(x)
    if mask.any():
        x[mask] = np.take(col_means, np.where(mask)[1])
    return x


def _norm_text(value: object) -> str:
    return str(value).strip().lower()


def diameter_subset(
    df: pd.DataFrame,
    *,
    param_set: Optional[str] = None,
    tip_middle_base_only: bool = False,
) -> pd.DataFrame:
    if "dataset" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    out = df[df["dataset"].map(_norm_text) == DIAMETER_DATASET].copy()
    if out.empty:
        return out

    if param_set is not None:
        if "param_set" not in out.columns:
            return pd.DataFrame(columns=df.columns)
        out = out[out["param_set"].map(_norm_text) == _norm_text(param_set)].copy()
        if out.empty:
            return out

    if tip_middle_base_only:
        if "channel" not in out.columns:
            return pd.DataFrame(columns=df.columns)
        out = out[out["channel"].map(_norm_text).isin(TIP_MIDDLE_BASE_CHANNELS)].copy()

    return out.reset_index(drop=True)


def averaged_tip_middle_base(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    d = diameter_subset(df, tip_middle_base_only=True)
    if d.empty or "channel" not in d.columns:
        return pd.DataFrame(columns=cols)

    var_cols = select_variable_columns(d)
    if not var_cols:
        return pd.DataFrame(columns=cols)

    # Average features across 1mm and 0.5mm for each channel (Tip/Middle/Base).
    grouped = d.groupby(d["channel"].map(_norm_text), dropna=False)
    rows = []
    for channel_key, g in grouped:
        row = {c: np.nan for c in cols}
        row["dataset"] = DIAMETER_DATASET
        row["param_set"] = "averaged_1.0mm0.5mm_tipmiddlebase"
        row["trial"] = np.nan
        row["channel"] = str(channel_key)
        row["sample_id"] = f"diameter__avg__{channel_key}"
        for c in var_cols:
            row[c] = pd.to_numeric(g[c], errors="coerce").mean()
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


def run_pca_block(df: pd.DataFrame, out_dir: Path, color_col: Optional[str] = None) -> int:
    if df.empty:
        print(f"Skipping {out_dir}: no rows")
        return 0

    var_cols = select_variable_columns(df)
    if not var_cols:
        print(f"Skipping {out_dir}: no numeric variables")
        return 0

    x = df[var_cols].to_numpy(dtype=float)
    x = impute_nan_matrix(x)

    if SCALE:
        means = np.nanmean(x, axis=0)
        stds = np.nanstd(x, axis=0, ddof=1)
        stds[stds == 0] = 1.0
        x = (x - means) / stds

    n_comp = min(N_COMPONENTS, x.shape[0], x.shape[1])
    if n_comp < 1:
        print(f"Skipping {out_dir}: insufficient shape for PCA ({x.shape[0]}x{x.shape[1]})")
        return 0

    pca = PCA(n_components=n_comp)
    try:
        scores = pca.fit_transform(x)
    except Exception as e:
        print(f"PCA failed for {out_dir}: {e}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    score_cols = [f"PC{i + 1}" for i in range(scores.shape[1])]

    meta_cols = [c for c in df.columns if c not in var_cols]
    scores_df = pd.DataFrame(scores, columns=score_cols)
    for c in meta_cols:
        scores_df[c] = df[c].values
    scores_df.to_csv(out_dir / "pca_scores.csv", index=False)

    pd.DataFrame(pca.components_.T, index=var_cols, columns=score_cols).to_csv(out_dir / "pca_loadings.csv")
    pd.DataFrame(
        {
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        },
        index=score_cols,
    ).to_csv(out_dir / "pca_explained_variance.csv")

    if scores.shape[1] >= 2:
        try:
            import matplotlib.pyplot as plt

            apply_publication_style()
            fig, ax = plt.subplots(figsize=(6.5, 6))
            if color_col and color_col in scores_df.columns:
                groups = list(scores_df.groupby(color_col, dropna=False))
                palette = get_palette(len(groups))
                for color, (label, grp) in zip(palette, groups):
                    ax.scatter(
                        grp["PC1"],
                        grp["PC2"],
                        label=str(label),
                        s=34,
                        color=color,
                        alpha=0.9,
                        edgecolors="#222222",
                        linewidths=0.35,
                    )
                ax.legend(loc="best", fontsize=8)
            else:
                ax.scatter(
                    scores_df["PC1"],
                    scores_df["PC2"],
                    s=34,
                    color=get_palette(1)[0],
                    alpha=0.9,
                    edgecolors="#222222",
                    linewidths=0.35,
                )

            evr = pca.explained_variance_ratio_
            pc1_var = 100.0 * float(evr[0]) if evr.size >= 1 else 0.0
            pc2_var = 100.0 * float(evr[1]) if evr.size >= 2 else 0.0
            ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.8, zorder=0)
            ax.axvline(0, color="#888888", linewidth=0.8, alpha=0.8, zorder=0)
            ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
            ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
            ax.set_title(f"PCA scores ({out_dir.name})")
            style_axes(ax, grid_axis="both")
            fig.tight_layout()
            fig.savefig(out_dir / "pca_scores.png", dpi=170)
            plt.close(fig)
        except ImportError:
            print("matplotlib not available, skipping score scatter plot")

    print(f"Wrote PCA results to {out_dir}")
    return 0


def main() -> int:
    if not IN_CSV.exists():
        print(f"Input file not found: {IN_CSV}. Run features.py first.")
        return 1

    df = pd.read_csv(IN_CSV)
    if df.empty:
        print(f"Input CSV is empty: {IN_CSV}")
        return 2

    status = 0
    status = max(status, run_pca_block(df, OUTPUT_ROOT / "meta" / "pca", color_col="dataset"))
    if "dataset" in df.columns:
        for dataset, g in df.groupby("dataset", dropna=False):
            dataset_name = str(dataset).strip()
            if dataset_name.lower() in {"air", "diameter"}:
                out_dir = OUTPUT_ROOT / dataset_name.lower() / "pca"
            else:
                out_dir = OUTPUT_ROOT / "meta" / "pca" / dataset_name
            status = max(status, run_pca_block(g.reset_index(drop=True), out_dir, color_col="param_set"))

    # Diameter-specific PCA blocks:
    # - keep existing 1mm and 0.5mm runs
    # - add averaged tip/middle/base run
    # - do not emit tip_middle_base* folders
    d_1mm = diameter_subset(df, param_set="1mm", tip_middle_base_only=False)
    status = max(status, run_pca_block(d_1mm, OUTPUT_ROOT / "diameter" / "pca" / "1mm", color_col="channel"))

    d_05mm = diameter_subset(df, param_set="0.5mm", tip_middle_base_only=False)
    status = max(status, run_pca_block(d_05mm, OUTPUT_ROOT / "diameter" / "pca" / "0.5mm", color_col="channel"))

    d_avg_tmb = averaged_tip_middle_base(df)
    status = max(
        status,
        run_pca_block(
            d_avg_tmb,
            OUTPUT_ROOT / "diameter" / "pca" / "averaged_1.0mm0.5mm_tipmiddlebase",
            color_col="channel",
        ),
    )

    return status


if __name__ == "__main__":
    raise SystemExit(main())
