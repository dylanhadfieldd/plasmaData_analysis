#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_STYLE_APPLIED = False
_ROMAN_STAGE = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


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
            font="STIXGeneral",
        )
    except Exception:
        pass

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "semibold",
            "axes.titlepad": 9.0,
            "axes.labelsize": 10.5,
            "axes.labelweight": "semibold",
            "axes.labelpad": 6.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.65,
            "grid.color": "#6b7280",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.fancybox": False,
            "legend.edgecolor": "#c4c4c4",
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.8,
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
    apply_publication_style()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out", width=0.8)
    ax.tick_params(axis="both", which="minor", direction="out", width=0.6)
    ax.grid(True, axis=grid_axis, which="major", alpha=0.2, linewidth=0.65)
    ax.grid(True, axis=grid_axis, which="minor", alpha=0.08, linewidth=0.5)
    ax.minorticks_on()


def spectral_interval_label(wavelengths: Sequence[float] | np.ndarray, decimals: int = 0) -> str:
    arr = np.asarray(wavelengths, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "[, nm]"
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if decimals <= 0:
        return f"[{int(round(lo))} , {int(round(hi))} nm]"
    return f"[{lo:.{decimals}f} , {hi:.{decimals}f} nm]"


def to_species_label(species: str) -> str:
    text = str(species).strip()
    if not text:
        return ""

    m_ion = re.fullmatch(r"([A-Za-z]{1,2})\s+([IVX]+)", text)
    if m_ion:
        elem = m_ion.group(1)
        stage = m_ion.group(2).upper()
        val = _ROMAN_STAGE.get(stage)
        if val is None or val <= 1:
            return elem
        charge = val - 1
        charge_txt = "+" if charge == 1 else f"{charge}+"
        return f"{elem}$^{{{charge_txt}}}$"

    m_mol = re.fullmatch(r"([A-Za-z]+)(\d*)([+-]*)", text.replace(" ", ""))
    if m_mol:
        base, digits, signs = m_mol.groups()
        out = base
        if digits:
            out += f"$_{digits}$"
        if signs:
            charge = signs[0]
            n = len(signs)
            charge_txt = charge if n == 1 else f"{n}{charge}"
            out += f"$^{{{charge_txt}}}$"
        return out

    return text


def species_labels(values: Iterable[object]) -> List[str]:
    return [to_species_label(str(v)) for v in values]
