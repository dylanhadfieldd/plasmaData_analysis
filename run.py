#!/usr/bin/env python3
from __future__ import annotations

from typing import Callable, List, Tuple

from analysis import compare, compose, features, ind_charts, kinetics, kinetics_charts, pca, preprocess, species

Step = Tuple[str, Callable[[], int]]

RUN_STEPS: List[Step] = [
    ("preprocess", preprocess.main),
    ("ind_charts", ind_charts.main),
    ("compose", compose.main),
    ("compare", compare.main),
    ("features", features.main),
    ("species", species.main),
    ("kinetics", kinetics.main),
    ("kinetics_charts", kinetics_charts.main),
    ("pca", pca.main),
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
