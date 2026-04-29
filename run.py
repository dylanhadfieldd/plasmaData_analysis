#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from analysis.output_paths import active_scopes, reset_active_scope_outputs, set_active_scopes


@dataclass(frozen=True)
class PipelineStep:
    name: str
    run: Callable[[], int]
    required_scope: str | None = None


def load_steps() -> list[PipelineStep]:
    from analysis import (
        chemical_modeling,
        chem_species_analysis,
        executive_reports,
        features,
        ms_output,
        pca,
        reaction_narrative,
        species,
    )
    from data_ingestion import preprocess
    from plots import air_reactive_auc, ms_output_charts, spectral_charts

    return [
        PipelineStep("preprocess", preprocess.main),
        PipelineStep("spectral_charts", spectral_charts.main),
        PipelineStep("ms_output", ms_output.main),
        PipelineStep("ms_output_charts", ms_output_charts.main),
        PipelineStep("features", features.main),
        PipelineStep("species", species.main),
        PipelineStep("chem_species_analysis", chem_species_analysis.main),
        PipelineStep("air_reactive_auc", air_reactive_auc.main, required_scope="air"),
        PipelineStep("pca", pca.main),
        PipelineStep("chemical_modeling", chemical_modeling.main),
        PipelineStep("reaction_narrative", reaction_narrative.main),
        PipelineStep("executive_reports", executive_reports.main),
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


def resolve_scopes(modes: Sequence[str]) -> tuple[str, ...]:
    parsed_modes: list[str] = []
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

    scopes: list[str] = []
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

    for step in run_steps:
        if step.required_scope and step.required_scope not in scope_set:
            print(f"\n== {step.name} (skipped: requires scope={step.required_scope}) ==")
            continue
        print(f"\n== {step.name} ==")
        code = step.run()
        if code != 0:
            print(f"\nStopped at {step.name} (exit code {code})")
            return code
    print("\nAll requested steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
