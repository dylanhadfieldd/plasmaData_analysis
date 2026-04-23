from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pandas as pd

from analysis.output_paths import SCOPES, metadata_csv_path


def scoped_slice(df: pd.DataFrame, scope: str, allow_global: bool = False) -> pd.DataFrame:
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
) -> List[Path]:
    written: List[Path] = []
    use_scopes = tuple(scopes) if scopes is not None else tuple(SCOPES)
    for scope in use_scopes:
        part = scoped_slice(df, scope, allow_global=allow_global)
        out_path = metadata_csv_path(scope, section, filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(out_path, index=False)
        written.append(out_path)
    return written

