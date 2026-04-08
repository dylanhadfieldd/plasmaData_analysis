#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

from analysis.output_paths import SCOPES

OUTPUT_ROOT = Path("output")
ALLOWED_SCOPE_DIRS = {"spectral", "chemspecies", "chemical_modeling", "pca", "metadata"}
SPECTRAL_CHEM_MODELING_NOISE_SUFFIXES = {".csv", ".xlsx", ".txt"}


def remove_any(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                next(path.iterdir())
            except StopIteration:
                try:
                    path.rmdir()
                except OSError:
                    pass


def clean_scope(scope: str) -> None:
    scope_dir = OUTPUT_ROOT / scope
    if not scope_dir.exists():
        return

    for item in scope_dir.iterdir():
        keep_dir = item.is_dir() and item.name in ALLOWED_SCOPE_DIRS
        keep_exec = item.is_file() and item.suffix.lower() == ".xlsx" and item.name.startswith(f"{scope}_executive_report")
        keep_temp_lock = item.is_file() and item.name.startswith("~$")
        if keep_dir or keep_exec or keep_temp_lock:
            continue
        remove_any(item)

    pca_root = scope_dir / "pca"
    if pca_root.exists():
        for item in pca_root.iterdir():
            keep = item.is_file() and item.suffix.lower() == ".png"
            if not keep:
                remove_any(item)

    for folder_name in ("spectral", "chemspecies", "chemical_modeling"):
        root = scope_dir / folder_name
        if not root.exists():
            continue
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SPECTRAL_CHEM_MODELING_NOISE_SUFFIXES:
                try:
                    file_path.unlink()
                except OSError:
                    pass
        prune_empty_dirs(root)


def clean_output_root_noise() -> None:
    if not OUTPUT_ROOT.exists():
        return
    for item in OUTPUT_ROOT.iterdir():
        if item.is_dir() and item.name in SCOPES:
            continue
        remove_any(item)


def main() -> int:
    if not OUTPUT_ROOT.exists():
        return 0

    clean_output_root_noise()
    for scope in SCOPES:
        clean_scope(scope)

    print(
        "Cleaned output layout to: output/*/{spectral,chemspecies,chemical_modeling,pca,metadata} "
        "+ *_executive_report.xlsx"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
