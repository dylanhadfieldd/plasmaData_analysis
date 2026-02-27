#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from analysis.plot_style import apply_publication_style, get_palette, style_axes

IN_SPECIES = Path("output/species_features.csv")
KINETICS_CFG = Path("configs/kinetics_species.csv")
TARGETS_CSV = Path("configs/experiments.csv")
OUT_CSV = Path("output/kinetics_features.csv")
OUT_SUMMARY_CSV = Path("output/kinetics_summary.csv")
MODEL_DIR = Path("stats/models")
EXPOSURE_TIME_S = 30.0
SAFE_TEXT_RE = re.compile(r"[^a-z0-9]+")

DEFAULT_CFG: List[Tuple[str, str, float, float, float]] = [
    ("OH_309", "oh_309_area", 1200.0, 0.80, 0.85),
    ("NO_gamma", "no_gamma_area", 350.0, 0.65, 0.70),
    ("N2_337", "n2_337_area", 900.0, 0.60, 0.50),
    ("N2plus_391", "n2plus_391_area", 1500.0, 0.85, 0.35),
    ("UVC_continuum", "uvc_continuum_area", 3000.0, 1.00, 0.15),
    ("Hgamma_434", "hgamma_434_area", 700.0, 0.20, 0.40),
]


def slug(text: str) -> str:
    out = SAFE_TEXT_RE.sub("_", str(text).strip().lower()).strip("_")
    return out or "species"


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(a / b)


def load_kinetics_config(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            DEFAULT_CFG,
            columns=[
                "species",
                "feature_col",
                "loss_rate_s",
                "sterilization_weight",
                "healing_weight",
            ],
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    needed = {"species", "feature_col", "loss_rate_s", "sterilization_weight", "healing_weight"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} must have columns: {', '.join(sorted(needed))}")
    return df[list(sorted(needed, key=lambda x: ["species", "feature_col", "loss_rate_s", "sterilization_weight", "healing_weight"].index(x)))]


def choose_scale(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    positive = x[x > 0]
    if not positive.empty:
        scale = float(positive.median())
        if np.isfinite(scale) and scale > 0:
            return scale
    xmax = float(np.nanmax(x.to_numpy(dtype=float))) if len(x) else float("nan")
    if np.isfinite(xmax) and xmax > 0:
        return xmax
    return 1.0


def species_dose_proxy(generation_norm: np.ndarray, loss_rate_s: float, exposure_time_s: float) -> np.ndarray:
    if not np.isfinite(loss_rate_s) or loss_rate_s <= 0:
        return np.full_like(generation_norm, np.nan, dtype=float)
    k = float(loss_rate_s)
    t = max(float(exposure_time_s), 0.0)
    steady = generation_norm / k
    dose = steady * (t - (1.0 - np.exp(-k * t)) / k)
    return dose


def build_experiment_sample_id(row: pd.Series) -> str:
    param_set = str(row.get("param_set", "")).strip()
    trial = row.get("trial")
    if not param_set or pd.isna(trial):
        return ""
    try:
        trial_int = int(float(trial))
        return f"{param_set}.{trial_int}"
    except Exception:
        return f"{param_set}.{trial}"


def write_dataset_csvs(df: pd.DataFrame, out_path: Path) -> None:
    if "dataset" not in df.columns:
        return
    for dataset, g in df.groupby("dataset", dropna=False):
        ds_dir = out_path.parent / str(dataset)
        ds_dir.mkdir(parents=True, exist_ok=True)
        g.to_csv(ds_dir / out_path.name, index=False)


def run_regression(kin_df: pd.DataFrame, model_dir: Path) -> int:
    if not TARGETS_CSV.exists():
        print(f"Targets file not found: {TARGETS_CSV}. Skipping regression model.")
        return 0

    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn not available, skipping regression model.")
        return 0

    targets = pd.read_csv(TARGETS_CSV)
    if not {"sample_id", "target"}.issubset(targets.columns):
        print(f"{TARGETS_CSV} must include columns: sample_id,target. Skipping regression model.")
        return 0

    data = kin_df.copy()
    data["experiment_sample_id"] = data.apply(build_experiment_sample_id, axis=1)
    labeled = data.merge(
        targets[["sample_id", "target"]].rename(columns={"sample_id": "experiment_sample_id"}),
        on="experiment_sample_id",
        how="inner",
    )

    if labeled.empty:
        print("No overlap between kinetics rows and target labels. Skipping regression model.")
        return 0

    ratio_cols = [
        "oh_to_n2_337_ratio",
        "n2_337_to_n2plus_391_ratio",
        "n2_sps_to_n2plus_ratio",
        "no_gamma_to_oh_ratio",
        "uvc_to_n2_337_ratio",
        "balmer_to_n2plus_ratio",
    ]
    dose_cols = sorted([c for c in labeled.columns if c.startswith("dose_")])
    index_cols = [c for c in ["sterilization_index", "healing_index", "healing_sterilization_ratio"] if c in labeled.columns]
    feature_cols = dose_cols + [c for c in ratio_cols if c in labeled.columns] + index_cols
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(labeled[c])]

    if not feature_cols:
        print("No numeric feature columns available for regression. Skipping model.")
        return 0

    model_df = labeled[["sample_id", "experiment_sample_id", "target"] + [c for c in ["dataset", "param_set", "trial", "channel"] if c in labeled.columns] + feature_cols].copy()
    x = model_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median()).fillna(0.0)
    y = pd.to_numeric(model_df["target"], errors="coerce")
    keep = y.notna()
    x = x.loc[keep].reset_index(drop=True)
    y = y.loc[keep].to_numpy(dtype=float)
    model_df = model_df.loc[keep].reset_index(drop=True)

    if len(model_df) < 4:
        print("Not enough labeled rows for regression (need >= 4). Skipping model.")
        return 0

    loo = LeaveOneOut()
    alpha_grid = np.logspace(-4, 4, 25)
    best_alpha = float(alpha_grid[0])
    best_rmse = float("inf")
    best_preds: np.ndarray | None = None

    for alpha in alpha_grid:
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=float(alpha))),
            ]
        )
        preds = cross_val_predict(pipe, x, y, cv=loo)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
            best_preds = preds

    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
    model.fit(x, y)
    pred_train = model.predict(x)
    pred_loo = best_preds if best_preds is not None else pred_train

    scaler = model.named_steps["scale"]
    ridge = model.named_steps["ridge"]
    coef_scaled = ridge.coef_.astype(float)
    scale = np.where(np.isfinite(scaler.scale_), scaler.scale_, 1.0)
    scale[scale == 0] = 1.0
    coef_raw = coef_scaled / scale
    intercept_raw = float(ridge.intercept_ - np.dot(scaler.mean_ / scale, coef_scaled))

    model_dir.mkdir(parents=True, exist_ok=True)

    coeff_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef_scaled": coef_scaled,
            "coef_raw_units": coef_raw,
            "alpha": best_alpha,
        }
    ).sort_values("coef_scaled", key=np.abs, ascending=False, ignore_index=True)
    coeff_df.to_csv(model_dir / "target_regression_coefficients.csv", index=False)

    pred_df = model_df.copy()
    pred_df["pred_train"] = pred_train
    pred_df["pred_loo"] = pred_loo
    pred_df["residual_train"] = pred_df["target"] - pred_df["pred_train"]
    pred_df["residual_loo"] = pred_df["target"] - pred_df["pred_loo"]
    pred_df.to_csv(model_dir / "target_regression_predictions.csv", index=False)

    metrics = pd.DataFrame(
        [
            ("n_samples", float(len(model_df))),
            ("n_features", float(len(feature_cols))),
            ("best_alpha", best_alpha),
            ("intercept_raw_units", intercept_raw),
            ("r2_train", float(r2_score(y, pred_train))),
            ("mae_train", float(mean_absolute_error(y, pred_train))),
            ("rmse_train", float(np.sqrt(mean_squared_error(y, pred_train)))),
            ("r2_loo", float(r2_score(y, pred_loo))),
            ("mae_loo", float(mean_absolute_error(y, pred_loo))),
            ("rmse_loo", float(np.sqrt(mean_squared_error(y, pred_loo)))),
        ],
        columns=["metric", "value"],
    )
    metrics.to_csv(model_dir / "target_regression_metrics.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        apply_publication_style()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        if "param_set" in model_df.columns:
            groups = list(model_df.groupby("param_set", dropna=False))
            palette = get_palette(len(groups))
            for color, (label, grp) in zip(palette, groups):
                idx = grp.index.to_numpy(dtype=int)
                ax.scatter(y[idx], pred_loo[idx], s=30, color=color, alpha=0.9, label=str(label))
            ax.legend(loc="best", fontsize=8)
        else:
            ax.scatter(y, pred_loo, s=30, color=get_palette(1)[0], alpha=0.9)
        lo = float(min(np.min(y), np.min(pred_loo)))
        hi = float(max(np.max(y), np.max(pred_loo)))
        if math.isfinite(lo) and math.isfinite(hi):
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black")
        ax.set_xlabel("Observed target")
        ax.set_ylabel("Predicted target (LOO)")
        ax.set_title("Target regression from species kinetics features")
        style_axes(ax, grid_axis="both")
        fig.tight_layout()
        fig.savefig(model_dir / "target_regression_scatter.png", dpi=180)
        plt.close(fig)
    except ImportError:
        print("matplotlib not available; skipped regression scatter plot.")

    print(f"Wrote {model_dir / 'target_regression_coefficients.csv'}")
    print(f"Wrote {model_dir / 'target_regression_predictions.csv'}")
    print(f"Wrote {model_dir / 'target_regression_metrics.csv'}")
    return 0


def main() -> int:
    if not IN_SPECIES.exists():
        print(f"Missing {IN_SPECIES}. Run species.py first.")
        return 1

    species_df = pd.read_csv(IN_SPECIES)
    if species_df.empty:
        print(f"Input CSV is empty: {IN_SPECIES}")
        return 2

    cfg = load_kinetics_config(KINETICS_CFG)
    out = species_df.copy()
    out["exposure_time_s"] = float(EXPOSURE_TIME_S)
    out["sterilization_index"] = 0.0
    out["healing_index"] = 0.0

    for _, row in cfg.iterrows():
        species = str(row["species"]).strip()
        species_slug = slug(species)
        feature_col = str(row["feature_col"]).strip()
        loss_rate = float(row["loss_rate_s"])
        w_ster = float(row["sterilization_weight"])
        w_heal = float(row["healing_weight"])

        if feature_col not in out.columns:
            print(f"[WARN] Missing feature column for {species}: {feature_col}. Skipping this species.")
            continue

        raw = pd.to_numeric(out[feature_col], errors="coerce").fillna(0.0)
        scale = choose_scale(raw)
        gen_norm = (raw / scale).to_numpy(dtype=float)
        c_ss = gen_norm / loss_rate if loss_rate > 0 else np.full_like(gen_norm, np.nan)
        dose = species_dose_proxy(gen_norm, loss_rate, EXPOSURE_TIME_S)

        out[f"gen_norm_{species_slug}"] = gen_norm
        out[f"c_ss_{species_slug}"] = c_ss
        out[f"dose_{species_slug}"] = dose
        out[f"norm_scale_{species_slug}"] = scale

        ster_comp = dose * w_ster
        heal_comp = dose * w_heal
        out[f"sterilization_component_{species_slug}"] = ster_comp
        out[f"healing_component_{species_slug}"] = heal_comp
        out["sterilization_index"] = pd.to_numeric(out["sterilization_index"], errors="coerce").fillna(0.0) + ster_comp
        out["healing_index"] = pd.to_numeric(out["healing_index"], errors="coerce").fillna(0.0) + heal_comp

    out["healing_sterilization_ratio"] = [
        safe_ratio(a, b)
        for a, b in zip(
            pd.to_numeric(out["healing_index"], errors="coerce"),
            pd.to_numeric(out["sterilization_index"], errors="coerce"),
        )
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    write_dataset_csvs(out, OUT_CSV)

    summary_metric_cols = [
        c
        for c in out.columns
        if c.startswith("dose_")
        or c.endswith("_index")
        or c in {"healing_sterilization_ratio"}
    ]
    summary = out.groupby(["dataset", "param_set", "channel"], dropna=False)[summary_metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["__".join([str(x) for x in col if str(x) != ""]).strip("__") for col in summary.columns]
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    write_dataset_csvs(summary, OUT_SUMMARY_CSV)

    print(f"Wrote {OUT_CSV} ({len(out)} rows)")
    print(f"Wrote {OUT_SUMMARY_CSV}")
    for dataset in sorted(out["dataset"].astype(str).unique()):
        print(f"Wrote {OUT_CSV.parent / dataset / OUT_CSV.name}")
        print(f"Wrote {OUT_SUMMARY_CSV.parent / dataset / OUT_SUMMARY_CSV.name}")

    return run_regression(out, MODEL_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
