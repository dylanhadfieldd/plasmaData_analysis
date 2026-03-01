#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.plot_style import apply_publication_style, get_palette, spectral_interval_label, style_axes

INPUT_LONG = Path("output/spectra_long.csv")
OUTPUT_DIR = Path("output/charts")
LOG_Y = False
NORMALIZE_Y = True
DPI = 200
SAFE_TEXT_RE = re.compile(r"[^a-z0-9]+")


def safe_name(text: str) -> str:
    out = SAFE_TEXT_RE.sub("_", text.strip().lower()).strip("_")
    return out or "plot"


def make_title(row: pd.Series) -> str:
    parts = [
        str(row.get("dataset", "")),
        str(row.get("param_set", "")),
        str(row.get("channel", "")),
    ]
    trial = row.get("trial")
    if pd.notna(trial):
        try:
            parts.append(f"trial {int(trial)}")
        except Exception:
            parts.append(f"trial {trial}")
    return " | ".join([p for p in parts if p])


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

    ok = 0
    bad = 0
    for sample_id, g in df.groupby("sample_id", dropna=False):
        g = g.sort_values("wavelength_nm")
        row = g.iloc[0]
        dataset = str(row["dataset"])
        plot_name = safe_name(str(sample_id)) + ".png"
        out_path = OUTPUT_DIR / dataset / plot_name

        try:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            color = get_palette(1)[0]
            x = g["wavelength_nm"].to_numpy(dtype=float)
            y = g["irradiance_W_m2_nm"].to_numpy(dtype=float)
            if NORMALIZE_Y:
                scale = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
                if scale > 0:
                    y = y / scale
                else:
                    y = np.zeros_like(y, dtype=float)
            ax.plot(x, y, color=color)
            ax.set_title(make_title(row))
            ax.set_xlabel(f"Wavelength {spectral_interval_label(x)}")
            ax.set_ylabel("Normalized Irradiance (a.u.)" if NORMALIZE_Y else "Irradiance (W m$^{-2}$ nm$^{-1}$)")
            style_axes(ax, grid_axis="both")
            if LOG_Y:
                ax.set_yscale("log")
            elif NORMALIZE_Y:
                ax.set_ylim(0, 1.05)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out_path, dpi=DPI)
            plt.close(fig)
            print(f"[OK] {sample_id} -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {sample_id}: {e}")
            bad += 1

    print(f"Done. OK={ok} FAIL={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
