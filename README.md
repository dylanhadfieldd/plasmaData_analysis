# Plasma Data Analysis

Run the full pipeline:

```bash
python run.py
```

Run a specific mode:

```bash
python run.py --mode air
python run.py --mode diameter
python run.py --mode meta
```

`--mode` also supports multiple values, for example:

```bash
python run.py --mode air diameter
```

The `-mode` alias is supported (for example `python run.py -mode air`).
Slash format is also accepted (for example `python run.py -mode air/diameter/meta`).

## Structure

- `data_ingestion/`: raw data loading, preprocessing pipeline, and NIST wire/fetch utilities
- `analysis/`: feature extraction, statistics, modeling, and report generation
- `analysis/ms_core.py`: shared spectral peak-detection/matching logic used by `analysis/ms_output.py`
- `plots/`: all chart/figure generation and plot styling
- `run.py`: single CLI orchestrator for the full pipeline

## Pipeline Stages

1. `preprocess`
2. `spectral_charts` (from `plots/`)
3. `ms_output`
4. `ms_output_charts` (from `plots/`)
5. `features`
6. `species`
7. `chem_species_analysis`
8. `air_reactive_auc` (runs only when `air` is part of mode)
9. `pca`
10. `chemical_modeling`
11. `reaction_narrative`
12. `executive_reports`

Selected scope outputs are reset at the start of each run, so no cleanup stage is required.

The pipeline also regenerates:

- `reaction wavelength narrative.md`
- `reaction wavelength narrative.pdf`

These narrative outputs are mode-dependent:

- `--mode air` uses air-only outputs.
- `--mode diameter` uses diameter-only outputs.
- `--mode all` (or both datasets) uses combined air+diameter outputs.

## Output Layout

- `output/<scope>/spectral/base/charts/{individual,composed,compared}`
- `output/<scope>/spectral/base/charts/diagnostics`
- `output/<scope>/spectral/labels`
- `output/<scope>/chemspecies/figures`
- `output/<scope>/chemical_modeling`
- `output/<scope>/pca`
- `output/<scope>/metadata/*`

`<scope>` is `air`, `diameter`, or `meta`.

All generated figure files use sequential names such as `Fig1.png`, `Fig2.png`, `Fig3.png`.
