#!/usr/bin/env python3

from __future__ import annotations

from typing import Callable, List, Tuple

from analysis import (
    chem_species_analysis,
    compare,
    compose,
    executive_reports,
    features,
    ind_charts,
    ms_output,
    ms_output_charts,
    pca,
    preprocess,
    species,
)

Step = Tuple[str, Callable[[], int]]

RUN_STEPS: List[Step] = [
    ("preprocess", preprocess.main),
    ("ind_charts", ind_charts.main),
    ("compose", compose.main),
    ("compare", compare.main),
    ("ms_output", ms_output.main),
    ("ms_output_charts", ms_output_charts.main),
    ("features", features.main),
    ("species", species.main),
    ("chem_species_analysis", chem_species_analysis.main),
    ("pca", pca.main),
    ("executive_reports", executive_reports.main),
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
