from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from analysis.output_paths import active_scopes, metadata_csv_path


def scoped_slice(df: pd.DataFrame, scope: str, allow_global: bool = False) -> pd.DataFrame:
    """Return the rows that should be written for a single output scope."""
    if scope == "meta":
        return df.copy()
    if "dataset" in df.columns:
        return df[df["dataset"].astype(str).str.lower() == scope].copy()
    if allow_global:
        return df.copy()
    return pd.DataFrame(columns=df.columns)


def write_scoped_csv(
    df: pd.DataFrame,
    section: str,
    filename: str,
    allow_global: bool = False,
    scopes: Sequence[str] | None = None,
) -> list[Path]:
    """Write a metadata CSV into each selected scope and return the paths."""
    written: list[Path] = []
    use_scopes = tuple(scopes) if scopes is not None else active_scopes()
    for scope in use_scopes:
        part = scoped_slice(df, scope, allow_global=allow_global)
        out_path = metadata_csv_path(scope, section, filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(out_path, index=False)
        written.append(out_path)
    return written

