# Plasma Data Analysis

This repo analyzes plasma spectra from `data/air` and `data/diameter` and generates figures, derived tables, and executive reports.

## Run

```bash
python run.py
```

## Output Layout

Each study scope (`air`, `diameter`, `meta`) is kept to the same structure:

```text
output/<scope>/
  spectral/                  # figures only
  chemspecies/               # figures only
  chemical_modeling/         # physics-model figures only
  pca/                       # pca_scores.png only
  metadata/                  # all CSV outputs (and scoped analysis metadata files)
  <scope>_executive_report.xlsx
```

The pipeline now enforces this layout and removes legacy/noisy output artifacts automatically.

## Pipeline Steps

`run.py` executes:

1. `preprocess`
2. `ind_charts`
3. `compose`
4. `compare`
5. `ms_output`
6. `ms_output_charts`
7. `features`
8. `species`
9. `chem_species_analysis`
10. `air_reactive_auc`
11. `pca`
12. `chemical_modeling`
13. `executive_reports`
14. `cleanup_outputs`
