# Plasma Data Analysis

This repository runs a scoped plasma spectroscopy analysis pipeline over the raw datasets in `data/`, writes regenerated figures and metadata under `output/`, and publishes the reaction narrative report under `docs/reports/`.

## Quick Start

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

```bash
python run.py
```

Run one or more scopes:

```bash
python run.py --mode air
python run.py --mode diameter
python run.py --mode air diameter
python run.py --mode air/diameter/meta
```

`--mode all` is the default and runs `air`, `diameter`, and `meta`. Selecting `air` or `diameter` also enables `meta` so combined outputs stay available.

## Project Layout

- `configs/`: analysis configuration tables such as species windows, target lines, and gas-condition metadata.
- `data/`: source spectra grouped by dataset.
- `data_ingestion/`: raw file discovery, parsing, NIST line retrieval, and preprocessing.
- `analysis/`: shared numeric utilities, scoped output helpers, feature extraction, PCA, species analysis, chemical modeling, and report assembly.
- `plots/`: chart-generation modules, publication styling, and figure helpers.
- `docs/reports/`: generated narrative report artifacts intended for review/sharing.
- `output/`: regenerated pipeline outputs by scope; safe to rebuild from source data and configs.

## Pipeline Stages

1. `preprocess`
2. `spectral_charts`
3. `ms_output`
4. `ms_output_charts`
5. `features`
6. `species`
7. `chem_species_analysis`
8. `air_reactive_auc` (only when `air` is active)
9. `pca`
10. `chemical_modeling`
11. `reaction_narrative`
12. `executive_reports`

Selected scope outputs are reset at the start of each run.

## Output Layout

Each scope is one of `air`, `diameter`, or `meta`.

- `output/<scope>/spectral/base/charts/{individual,composed,compared}`
- `output/<scope>/spectral/base/charts/diagnostics`
- `output/<scope>/spectral/labels`
- `output/<scope>/chemspecies/figures`
- `output/<scope>/chemical_modeling`
- `output/<scope>/pca`
- `output/<scope>/metadata/*`

Generated figures use sequential names such as `Fig1.png`, `Fig2.png`, and `Fig3.png`.

## Reports

The reaction narrative stage writes:

- `docs/reports/reaction_wavelength_narrative.md`
- `docs/reports/reaction_wavelength_narrative.pdf`

Those files are mode-dependent: `--mode air` uses air-only outputs, `--mode diameter` uses diameter-only outputs, and `--mode all` uses combined air+diameter outputs.
