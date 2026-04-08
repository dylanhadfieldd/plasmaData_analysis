#!/usr/bin/env python3

from __future__ import annotations

from typing import Callable, List, Tuple

from analysis import (
    air_reactive_auc,
    chemical_modeling,
    chem_species_analysis,
    cleanup_outputs,
    executive_reports,
    features,
    ms_output,
    ms_output_charts,
    pca,
    preprocess,
    spectral_charts,
    species,
)

Step = Tuple[str, Callable[[], int]]

RUN_STEPS: List[Step] = [
    ("preprocess", preprocess.main),
    ("spectral_charts", spectral_charts.main),
    ("ms_output", ms_output.main),
    ("ms_output_charts", ms_output_charts.main),
    ("features", features.main),
    ("species", species.main),
    ("chem_species_analysis", chem_species_analysis.main),
    ("air_reactive_auc", air_reactive_auc.main),
    ("pca", pca.main),
    ("chemical_modeling", chemical_modeling.main),
    ("executive_reports", executive_reports.main),
    ("cleanup_outputs", cleanup_outputs.main),
]


def main() -> int:
    for name, fn in RUN_STEPS:
        print(f"\n== {name} ==")
        code = fn()
        if code != 0:
            print(f"\nStopped at {name} (exit code {code})")
            return code
    print("\nAll steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
