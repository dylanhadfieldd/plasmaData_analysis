#!/usr/bin/env python3
from __future__ import annotations

from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt

_STYLE_APPLIED = False


def apply_publication_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    try:
        import seaborn as sns

        sns.set_theme(
            context="paper",
            style="whitegrid",
            palette="colorblind",
            font="DejaVu Sans",
        )
    except Exception:
        pass

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.2,
        }
    )
    _STYLE_APPLIED = True


def get_palette(n: int, name: str = "colorblind") -> List[object]:
    apply_publication_style()
    if n <= 0:
        return []
    try:
        import seaborn as sns

        return list(sns.color_palette(name, n))
    except Exception:
        cmap_name = "tab20" if n > 10 else "tab10"
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i % cmap.N) for i in range(n)]


def style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.22, linewidth=0.7)
