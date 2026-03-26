from __future__ import annotations

from pathlib import Path

OUTPUT_ROOT = Path("output")
SCOPES = ("air", "diameter", "meta")


def scope_root(scope: str) -> Path:
    return OUTPUT_ROOT / scope


def spectral_dir(scope: str) -> Path:
    return scope_root(scope) / "spectral"


def chemspecies_dir(scope: str) -> Path:
    return scope_root(scope) / "chemspecies"


def chemical_modeling_dir(scope: str) -> Path:
    return scope_root(scope) / "chemical_modeling"


def pca_dir(scope: str) -> Path:
    return scope_root(scope) / "pca"


def metadata_dir(scope: str) -> Path:
    return scope_root(scope) / "metadata"


def metadata_section_dir(scope: str, section: str) -> Path:
    return metadata_dir(scope) / section


def metadata_csv_path(scope: str, section: str, filename: str) -> Path:
    return metadata_section_dir(scope, section) / filename


def spectral_individual_dir(scope: str) -> Path:
    return spectral_dir(scope) / "base" / "charts" / "individual"


def spectral_composed_dir(scope: str) -> Path:
    return spectral_dir(scope) / "base" / "charts" / "composed"


def spectral_compared_dir(scope: str) -> Path:
    return spectral_dir(scope) / "base" / "charts" / "compared"


def spectral_diagnostics_dir(scope: str) -> Path:
    return spectral_dir(scope) / "base" / "charts" / "diagnostics"


def spectral_labels_dir(scope: str) -> Path:
    return spectral_dir(scope) / "labels"


def chemspecies_figures_dir(scope: str) -> Path:
    return chemspecies_dir(scope) / "figures"


def ensure_scope_layout(scope: str) -> None:
    dirs = (
        spectral_individual_dir(scope),
        spectral_composed_dir(scope),
        spectral_compared_dir(scope),
        spectral_diagnostics_dir(scope),
        spectral_labels_dir(scope),
        chemspecies_figures_dir(scope),
        chemical_modeling_dir(scope),
        pca_dir(scope),
        metadata_dir(scope),
    )
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def ensure_all_scope_layouts() -> None:
    for scope in SCOPES:
        ensure_scope_layout(scope)
