#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from typing import Callable, List, Sequence, Tuple

from analysis.output_paths import active_scopes, reset_active_scope_outputs, set_active_scopes

Step = Tuple[str, Callable[[], int], str | None]


def load_steps() -> List[Step]:
    from analysis import (
        chemical_modeling,
        chem_species_analysis,
        executive_reports,
        features,
        ms_output,
        pca,
        preprocess,
        reaction_narrative,
        species,
    )
    from plots import air_reactive_auc, ms_output_charts, spectral_charts

    return [
        ("preprocess", preprocess.main, None),
        ("spectral_charts", spectral_charts.main, None),
        ("ms_output", ms_output.main, None),
        ("ms_output_charts", ms_output_charts.main, None),
        ("features", features.main, None),
        ("species", species.main, None),
        ("chem_species_analysis", chem_species_analysis.main, None),
        ("air_reactive_auc", air_reactive_auc.main, "air"),
        ("pca", pca.main, None),
        ("chemical_modeling", chemical_modeling.main, None),
        ("reaction_narrative", reaction_narrative.main, None),
        ("executive_reports", executive_reports.main, None),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run plasma analysis pipeline.")
    parser.add_argument(
        "-mode",
        "--mode",
        nargs="*",
        default=["all"],
        help="Scope mode(s): all, air, diameter, meta. Supports slash/comma forms like air/diameter.",
    )
    return parser.parse_args(argv)


def resolve_scopes(modes: Sequence[str]) -> Tuple[str, ...]:
    parsed_modes: List[str] = []
    raw_modes = list(modes or ["all"])
    for token in raw_modes:
        parts = [p.strip().lower() for p in re.split(r"[,/]+", str(token)) if p.strip()]
        parsed_modes.extend(parts)

    if not parsed_modes:
        parsed_modes = ["all"]

    valid_modes = {"all", "air", "diameter", "meta"}
    invalid = sorted({m for m in parsed_modes if m not in valid_modes})
    if invalid:
        raise ValueError(f"Invalid mode value(s): {', '.join(invalid)}. Use all, air, diameter, or meta.")

    selected = set(parsed_modes)
    if not selected or "all" in selected:
        return ("air", "diameter", "meta")

    include_air = "air" in selected
    include_diameter = "diameter" in selected
    include_meta = "meta" in selected or include_air or include_diameter

    scopes: List[str] = []
    if include_air:
        scopes.append("air")
    if include_diameter:
        scopes.append("diameter")
    if include_meta:
        scopes.append("meta")
    return tuple(scopes) or ("air", "diameter", "meta")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as e:
        return int(e.code)

    try:
        scopes = resolve_scopes(args.mode)
    except ValueError as e:
        print(str(e))
        return 2

    set_active_scopes(scopes)
    reset_active_scope_outputs()

    scope_set = set(active_scopes())
    print(f"Modes: {', '.join(args.mode) if args.mode else 'all'}")
    print(f"Active scopes: {', '.join(active_scopes())}")

    try:
        run_steps = load_steps()
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        print(f"Missing dependency: {missing}. Install project requirements and re-run.")
        return 2

    for name, fn, required_scope in run_steps:
        if required_scope and required_scope not in scope_set:
            print(f"\n== {name} (skipped: requires scope={required_scope}) ==")
            continue
        print(f"\n== {name} ==")
        code = fn()
        if code != 0:
            print(f"\nStopped at {name} (exit code {code})")
            return code
    print("\nAll requested steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
