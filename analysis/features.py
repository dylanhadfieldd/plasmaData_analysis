#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LONG_CSV = Path("output/spectra_long.csv")
OUT_CSV = Path("output/features.csv")
BANDS_CSV = Path("configs/wavelengths.csv")
NORMALIZE = "none"
DEFAULT_BANDS: List[Tuple[str, float, float]] = [
    ("UVC", 200.0, 280.0),
    ("UVB", 280.0, 315.0),
    ("UVA", 315.0, 400.0),
]


def load_bands(config_path: Path) -> List[Tuple[str, float, float]]:
    if not config_path.exists():
        return DEFAULT_BANDS

    df = pd.read_csv(config_path)
    needed = {"band", "start_nm", "end_nm"}
    if not needed.issubset(set(c.lower() for c in df.columns)):
        raise ValueError(f"{config_path} must have columns: band,start_nm,end_nm")

    df = df.rename(columns={c: c.lower() for c in df.columns})
    bands: List[Tuple[str, float, float]] = []
    for _, r in df.iterrows():
        band = str(r["band"]).strip()
        start = float(r["start_nm"])
        end = float(r["end_nm"])
        if end <= start:
            raise ValueError(f"Band {band} has end_nm <= start_nm ({start}, {end})")
        bands.append((band, start, end))
    return bands


def trapz_integral(wl: np.ndarray, y: np.ndarray) -> float:
    if wl.size < 2:
        return float("nan")
    return float(np.trapz(y, wl))


def centroid(wl: np.ndarray, y: np.ndarray) -> float:
    area = float(np.trapz(y, wl))
    if not np.isfinite(area) or area <= 0:
        return float("nan")
    return float(np.trapz(wl * y, wl) / area)


def band_integral(wl: np.ndarray, y: np.ndarray, start: float, end: float) -> float:
    mask = (wl >= start) & (wl <= end)
    if mask.sum() < 2:
        return 0.0
    return trapz_integral(wl[mask], y[mask])


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(a / b)


def normalize_curve(wl: np.ndarray, y: np.ndarray) -> np.ndarray:
    mode = NORMALIZE.lower()
    if mode == "none":
        return y
    if mode == "area":
        area = trapz_integral(wl, y)
        return y / area if np.isfinite(area) and area != 0 else y
    if mode == "max":
        mx = float(np.nanmax(y)) if y.size else float("nan")
        return y / mx if np.isfinite(mx) and mx != 0 else y
    raise ValueError(f"Unknown NORMALIZE mode: {NORMALIZE}")


def write_dataset_features(features_df: pd.DataFrame, out_root: Path) -> None:
    for dataset, g in features_df.groupby("dataset", dropna=False):
        ds_dir = out_root / str(dataset)
        ds_dir.mkdir(parents=True, exist_ok=True)
        g.to_csv(ds_dir / "features.csv", index=False)


def main() -> int:
    if not LONG_CSV.exists():
        print(f"Missing {LONG_CSV}. Run preprocess.py first.")
        return 1

    bands = load_bands(BANDS_CSV)
    df = pd.read_csv(LONG_CSV)

    required = {"dataset", "sample_id", "param_set", "channel", "wavelength_nm", "irradiance_W_m2_nm"}
    if not required.issubset(df.columns):
        print(f"{LONG_CSV} must have columns: {sorted(required)}")
        return 2

    group_cols = ["sample_id"] + [c for c in ["dataset", "param_set", "trial", "channel"] if c in df.columns]
    rows: List[Dict[str, object]] = []

    for key, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("wavelength_nm")
        wl = g["wavelength_nm"].to_numpy(dtype=float)
        y = normalize_curve(wl, g["irradiance_W_m2_nm"].to_numpy(dtype=float))

        total = trapz_integral(wl, y)
        c_nm = centroid(wl, y)
        if y.size:
            i_peak = int(np.nanargmax(y))
            peak_wl = float(wl[i_peak])
            peak_y = float(y[i_peak])
        else:
            peak_wl = float("nan")
            peak_y = float("nan")

        band_vals = {band: band_integral(wl, y, start, end) for band, start, end in bands}
        bmap = {b.upper(): v for b, v in band_vals.items()}

        out: Dict[str, object] = {}
        if isinstance(key, tuple):
            for col, val in zip(group_cols, key):
                out[col] = val
        else:
            out[group_cols[0]] = key

        out.update(
            {
                "normalize": NORMALIZE,
                "total_irradiance": total,
                "centroid_nm": c_nm,
                "peak_wavelength_nm": peak_wl,
                "peak_irradiance": peak_y,
            }
        )

        for band, _, _ in bands:
            out[f"{band}_integral"] = band_vals[band]
            out[f"{band}_frac"] = safe_ratio(band_vals[band], total)

        out["uva_uvb_ratio"] = safe_ratio(bmap.get("UVA", float("nan")), bmap.get("UVB", float("nan")))
        out["uvb_uvc_ratio"] = safe_ratio(bmap.get("UVB", float("nan")), bmap.get("UVC", float("nan")))
        out["uva_uvc_ratio"] = safe_ratio(bmap.get("UVA", float("nan")), bmap.get("UVC", float("nan")))
        rows.append(out)

    out_df = pd.DataFrame(rows).sort_values(["dataset", "sample_id"], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    write_dataset_features(out_df, OUT_CSV.parent)

    print(f"Wrote {OUT_CSV} ({len(out_df)} rows)")
    for dataset in sorted(out_df["dataset"].astype(str).unique()):
        print(f"Wrote {OUT_CSV.parent / dataset / 'features.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
