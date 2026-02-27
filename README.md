# Plasma Data Analysis 

A lightweight Python project for processing and analyzing experimental plasma data. Scripts for preprocessing, feature extraction, statistical analysis, and visualizations.

## Summary

This workspace organizes data and tools to:

1. **Ingest raw CSV files** from `data/raw` into a clean format.
2. **Preprocess** and transform measurements using `preprocess.py`.
3. **Extract features** and compute statistics via `features.py` and `stats/` modules.
4. **Extract species-aware spectral features** (line/band windows and ratios) with `species.py`.
5. **Apply a chemical-engineering kinetics proxy layer** (steady-state and dose proxies + target regression) with `kinetics.py`.
6. **Generate kinetics visual analytics** (dataset-level and meta-analysis figures) with `kinetics_charts.py`.
7. **Perform comparative analyses** and dimensionality reduction (PCA) in `analysis/`.
8. **Generate charts** and combined summaries in `output/`.

## Workflow

1. Place raw CSV files in `data/raw`.
2. Run `preprocess.py` to clean and structure the data.
3. Execute `features.py` to derive relevant metrics.
4. Execute `species.py` to extract species windows (for example OH, N2, N2+) and line/band ratios.
5. Execute `kinetics.py` to compute kinetics-inspired concentration/dose proxies and fit a target regression model (when labels are available in `configs/experiments.csv`).
6. Execute `kinetics_charts.py` to render kinetics figures for each dataset and cross-dataset meta-analysis.
7. Use `analysis/` scripts (`compare.py`, `compose.py`, `pca.py`) to explore and compare datasets.
8. Review outputs in `output/charts`, `output/combined`, `output/species_*.csv`, `output/kinetics_*.csv`, `output/kinetics_charts`, and `stats/models`.


## Requirements

- Python 3.x
- pandas, numpy, matplotlib (or other dependencies defined in scripts)

## Directory Structure

```
analysis/       # analysis tools
configs/        # experiment configuration files
data/           # raw and processed datasets
output/         # generated charts & combined results
stats/          # statistical utilities
```

## New Config Files

- `configs/species_windows.csv`: configurable spectral windows for species-oriented feature extraction.
- `configs/kinetics_species.csv`: loss rates and weighting factors for kinetics-dose proxy indices.

---

