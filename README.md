# Plasma Data Analysis

Run the full pipeline with:

```bash
python run.py
```

## Pipeline

1. `preprocess`
2. `spectral_charts`
3. `ms_output`
4. `ms_output_charts`
5. `features`
6. `species`
7. `chem_species_analysis`
8. `air_reactive_auc`
9. `pca`
10. `chemical_modeling`
11. `executive_reports`
12. `cleanup_outputs`

## Output Layout

- `output/<scope>/spectral/base/charts/{individual,composed,compared}`
- `output/<scope>/spectral/base/charts/diagnostics`
- `output/<scope>/spectral/labels`
- `output/<scope>/chemspecies/figures`
- `output/<scope>/chemical_modeling`
- `output/<scope>/pca`
- `output/<scope>/metadata/*`

`<scope>` is `air`, `diameter`, or `meta`.

All generated figure files now use simple sequential names: `Fig1.png`, `Fig2.png`, `Fig3.png`, etc.
