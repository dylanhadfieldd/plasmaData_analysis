from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def clear_figure_files(path: Path, patterns: Iterable[str] = ("*.png", "*.svg")) -> None:
    """Ensure a figure directory exists and remove stale generated image files."""
    path.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for old in path.glob(pattern):
            old.unlink()


def write_message_figure(
    out_path: Path,
    message: str,
    figsize: tuple[float, float] = (8.0, 4.0),
    dpi: int = 220,
) -> Path:
    """Write a small placeholder figure explaining why data-driven output is empty."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path

