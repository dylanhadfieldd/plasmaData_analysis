#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

IN_LONG = Path("output/spectra_long.csv")
WINDOWS_CSV = Path("configs/species_windows.csv")
OUT_CSV = Path("output/species_features.csv")
OUT_SUMMARY_CSV = Path("output/species_summary.csv")
SAFE_TEXT_RE = re.compile(r"[^a-z0-9]+")

REQUIRED_COLS = {"sample_id", "wavelength_nm", "irradiance_W_m2_nm"}
DEFAULT_WINDOWS: List[Tuple[str, float, float, str]] = [
    ("NO_gamma", 220.0, 250.0, "band"),
    ("OH_309", 306.0, 312.0, "line"),
    ("N2_315", 313.0, 318.0, "band"),
    ("N2_337", 334.0, 340.0, "line"),
    ("N2_357", 354.0, 360.0, "line"),
    ("N2_380", 377.0, 383.0, "line"),
    ("N2plus_391", 388.0, 394.0, "line"),
    ("N2plus_427", 424.0, 430.0, "line"),
    ("Hgamma_434", 431.0, 437.0, "line"),
    ("Hbeta_486", 483.0, 489.0, "line"),
    ("UVC_continuum", 200.0, 280.0, "band"),
]


def slug(text: str) -> str:
    out = SAFE_TEXT_RE.sub("_", str(text).strip().lower()).strip("_")
    return out or "window"


def trapz_integral(wl: np.ndarray, y: np.ndarray) -> float:
    if wl.size < 2:
        return 0.0
    return float(np.trapz(y, wl))


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(a / b)


def load_windows(config_path: Path) -> List[Dict[str, object]]:
    if not config_path.exists():
        return [
            {
                "species": species,
                "species_slug": slug(species),
                "start_nm": start,
                "end_nm": end,
                "kind": kind,
            }
            for species, start, end, kind in DEFAULT_WINDOWS
        ]

    df = pd.read_csv(config_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    needed = {"species", "start_nm", "end_nm"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{config_path} must have columns: species,start_nm,end_nm")

    windows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        species = str(row["species"]).strip()
        start = float(row["start_nm"])
        end = float(row["end_nm"])
        kind = str(row.get("kind", "band")).strip().lower() or "band"
        if end <= start:
            raise ValueError(f"Invalid window for {species}: end_nm <= start_nm ({start}, {end})")
        windows.append(
            {
                "species": species,
                "species_slug": slug(species),
                "start_nm": start,
                "end_nm": end,
                "kind": kind,
            }
        )
    return windows


def extract_window_metrics(wl: np.ndarray, y: np.ndarray, start_nm: float, end_nm: float) -> Dict[str, float]:
    mask = (wl >= start_nm) & (wl <= end_nm)
    if int(mask.sum()) == 0:
        return {"area": 0.0, "peak": float("nan"), "peak_nm": float("nan"), "n_points": 0.0}

    wl_win = wl[mask]
    y_win = y[mask]
    area = trapz_integral(wl_win, y_win)
    peak_idx = int(np.nanargmax(y_win)) if y_win.size else 0
    peak = float(y_win[peak_idx]) if y_win.size else float("nan")
    peak_nm = float(wl_win[peak_idx]) if wl_win.size else float("nan")
    return {"area": area, "peak": peak, "peak_nm": peak_nm, "n_points": float(y_win.size)}


def add_grouped_species_features(row: Dict[str, object]) -> None:
    n2_sps = (
        float(row.get("n2_315_area", 0.0))
        + float(row.get("n2_337_area", 0.0))
        + float(row.get("n2_357_area", 0.0))
        + float(row.get("n2_380_area", 0.0))
    )
    n2_plus = float(row.get("n2plus_391_area", 0.0)) + float(row.get("n2plus_427_area", 0.0))
    balmer = float(row.get("hgamma_434_area", 0.0)) + float(row.get("hbeta_486_area", 0.0))

    row["n2_sps_total_area"] = n2_sps
    row["n2plus_total_area"] = n2_plus
    row["balmer_total_area"] = balmer

    row["oh_to_n2_337_ratio"] = safe_ratio(float(row.get("oh_309_area", float("nan"))), float(row.get("n2_337_area", float("nan"))))
    row["n2_337_to_n2plus_391_ratio"] = safe_ratio(
        float(row.get("n2_337_area", float("nan"))), float(row.get("n2plus_391_area", float("nan")))
    )
    row["n2_sps_to_n2plus_ratio"] = safe_ratio(n2_sps, n2_plus)
    row["no_gamma_to_oh_ratio"] = safe_ratio(float(row.get("no_gamma_area", float("nan"))), float(row.get("oh_309_area", float("nan"))))
    row["uvc_to_n2_337_ratio"] = safe_ratio(
        float(row.get("uvc_continuum_area", float("nan"))), float(row.get("n2_337_area", float("nan")))
    )
    row["balmer_to_n2plus_ratio"] = safe_ratio(balmer, n2_plus)


def write_dataset_csvs(df: pd.DataFrame, out_path: Path) -> None:
    for dataset, g in df.groupby("dataset", dropna=False):
        ds_dir = out_path.parent / str(dataset)
        ds_dir.mkdir(parents=True, exist_ok=True)
        g.to_csv(ds_dir / out_path.name, index=False)


def main() -> int:
    if not IN_LONG.exists():
        print(f"Missing {IN_LONG}. Run preprocess.py first.")
        return 1

    spectra = pd.read_csv(IN_LONG)
    if not REQUIRED_COLS.issubset(spectra.columns):
        print(f"{IN_LONG} must have columns: {sorted(REQUIRED_COLS)}")
        return 2

    windows = load_windows(WINDOWS_CSV)
    id_cols = ["sample_id"] + [c for c in ["dataset", "param_set", "trial", "channel"] if c in spectra.columns]
    rows: List[Dict[str, object]] = []

    for key, g in spectra.groupby(id_cols, dropna=False):
        g = g.sort_values("wavelength_nm")
        wl = g["wavelength_nm"].to_numpy(dtype=float)
        y = g["irradiance_W_m2_nm"].to_numpy(dtype=float)
        total = trapz_integral(wl, y)

        row: Dict[str, object] = {}
        if isinstance(key, tuple):
            for col, val in zip(id_cols, key):
                row[col] = val
        else:
            row[id_cols[0]] = key
        row["total_integral"] = total
        row["n_windows"] = len(windows)

        for w in windows:
            stem = str(w["species_slug"])
            metrics = extract_window_metrics(wl, y, float(w["start_nm"]), float(w["end_nm"]))
            row[f"{stem}_area"] = metrics["area"]
            row[f"{stem}_peak"] = metrics["peak"]
            row[f"{stem}_peak_nm"] = metrics["peak_nm"]
            row[f"{stem}_frac_total"] = safe_ratio(metrics["area"], total)
            row[f"{stem}_points"] = metrics["n_points"]

        add_grouped_species_features(row)
        rows.append(row)

    if not rows:
        print("No rows produced from species extraction.")
        return 2

    features_df = pd.DataFrame(rows).sort_values(["dataset", "sample_id"], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUT_CSV, index=False)
    write_dataset_csvs(features_df, OUT_CSV)

    summary_cols = [c for c in features_df.columns if c.endswith("_area") or c.endswith("_ratio")]
    summary = (
        features_df.groupby(["dataset", "param_set", "channel"], dropna=False)[summary_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["__".join([str(x) for x in col if str(x) != ""]).strip("__") for col in summary.columns]
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    write_dataset_csvs(summary, OUT_SUMMARY_CSV)

    print(f"Wrote {OUT_CSV} ({len(features_df)} rows)")
    print(f"Wrote {OUT_SUMMARY_CSV}")
    for dataset in sorted(features_df["dataset"].astype(str).unique()):
        print(f"Wrote {OUT_CSV.parent / dataset / OUT_CSV.name}")
        print(f"Wrote {OUT_SUMMARY_CSV.parent / dataset / OUT_SUMMARY_CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
