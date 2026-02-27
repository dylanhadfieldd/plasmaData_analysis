#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from analysis.plot_style import apply_publication_style, get_palette, style_axes

INPUT_LONG = Path("output/spectra_long.csv")
OUTPUT_DIR = Path("output/composed")
LOG_Y = False
DPI = 200
SAFE_TEXT_RE = re.compile(r"[^a-z0-9]+")


def safe_name(text: str) -> str:
    out = SAFE_TEXT_RE.sub("_", text.strip().lower()).strip("_")
    return out or "group"


def label_curve(row: pd.Series) -> str:
    parts: List[str] = []
    channel = str(row.get("channel", "")).strip()
    if channel and channel.lower() not in {"bulk", "irradiance"}:
        parts.append(channel)
    trial = row.get("trial")
    if pd.notna(trial):
        try:
            parts.append(f"trial {int(trial)}")
        except Exception:
            parts.append(f"trial {trial}")
    if not parts:
        parts.append(str(row.get("sample_id", "sample")))
    return " | ".join(parts)


def plot_group(dataset: str, param_set: str, group: pd.DataFrame, out_path: Path) -> None:
    grouped = list(group.groupby("sample_id", dropna=False))
    palette = get_palette(len(grouped))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for color, (sample_id, g) in zip(palette, grouped):
        g = g.sort_values("wavelength_nm")
        row = g.iloc[0]
        ax.plot(
            g["wavelength_nm"].to_numpy(),
            g["irradiance_W_m2_nm"].to_numpy(),
            label=label_curve(row),
            color=color,
            alpha=0.95,
        )

    ax.set_title(f"{dataset} | {param_set}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Irradiance (W/(m2*nm))")
    style_axes(ax, grid_axis="both")
    if LOG_Y:
        ax.set_yscale("log")
    if group["sample_id"].nunique() > 1:
        ax.legend(loc="best", fontsize=9, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main() -> int:
    apply_publication_style()

    if not INPUT_LONG.exists():
        print(f"Missing {INPUT_LONG}. Run preprocess.py first.")
        return 1

    df = pd.read_csv(INPUT_LONG)
    required = {"dataset", "sample_id", "param_set", "wavelength_nm", "irradiance_W_m2_nm"}
    if not required.issubset(df.columns):
        print(f"{INPUT_LONG} must have columns: {sorted(required)}")
        return 2

    ok = 0
    bad = 0
    grouped = df.groupby(["dataset", "param_set"], dropna=False)
    for (dataset, param_set), g in grouped:
        dataset = str(dataset)
        param_set = str(param_set)
        out_path = OUTPUT_DIR / dataset / f"{safe_name(param_set)}.png"
        try:
            plot_group(dataset, param_set, g, out_path)
            print(f"[OK] {dataset}/{param_set} ({g['sample_id'].nunique()} curves) -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {dataset}/{param_set}: {e}")
            bad += 1

    print(f"Done. Groups OK={ok} FAIL={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
