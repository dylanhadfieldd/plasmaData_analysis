# Plasma Data Analysis

Short summary: this pipeline takes raw plasma spectra CSVs and produces scoped analysis outputs for `air`, `diameter`, and `meta`.

## Input

- Raw spectra CSV files in `data/raw/`
- Target species list in `configs/target_species_lines.csv`
- NIST species query list in `configs/nist_fetch_species.csv`
- Optional offline NIST mirror: `configs/nist_lines.csv` (template: `configs/nist_lines_template.csv`)

## Run

```bash
python run.py
```

## Output

Pipeline writes results to:

- `output/air/`
- `output/diameter/`
- `output/meta/`

Each scope contains:

- `spectral/`
- `spectral/base/raw` (traceable CSV outputs)
- `spectral/base/charts` (unlabeled spectral figures)
- `spectral/labels` (species-labeled spectral figures)
- `chemspecies/csv` and `chemspecies/figures`
- `pca/`
- `<scope>_executive_report.xlsx`

