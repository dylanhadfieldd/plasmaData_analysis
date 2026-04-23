#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.output_paths import chemspecies_figures_dir, ensure_all_scope_layouts, metadata_csv_path
from plots.style import apply_publication_style, style_axes


CSV_OUT = metadata_csv_path("air", "chemspecies", "air_species_auc_normalized.csv")
FIG_ROOT = chemspecies_figures_dir("air") / "air_reactive_auc"

AIR_LONG_CSV = metadata_csv_path("air", "spectral", "spectra_long.csv")
AIR_RAW_DIR = Path("data") / "air"

AIR_LEVEL_MAP: Dict[str, str] = {
    "100H": "none",
    "100H.cont.trial": "none",
    "5H..01A": "low",
    "5H..5A": "medium",
    "5H..9A": "high",
    "1H.1A-CONT": "medium",
    "1H.9A": "high",
}
AIR_LEVEL_ORDER = ["none", "low", "medium", "high"]
AIR_LEVEL_DISPLAY: Dict[str, str] = {
    "none": "No Air (100H)",
    "low": "Low Air",
    "medium": "Medium Air",
    "high": "High Air",
}

SPECIES_WINDOWS: Dict[str, tuple[float, float]] = {
    "N2": (334.0, 340.0),
    "N2+": (388.0, 394.0),
    "OH": (306.0, 312.0),
}
SPECIES_ORDER = ["N2", "N2+", "OH"]
GROUP2_SPECIES_ORDER = ["N2", "N2+", "OH"]
SPECIES_COLORS: Dict[str, str] = {
    "OH": "#1b9e77",
    "N2": "#d95f02",
    "N2+": "#7570b3",
}

PNG_DPI = 600


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def prepare_output_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for stale_dir in (FIG_ROOT / "group1_auc_vs_species", FIG_ROOT / "group2_auc_vs_air_input"):
        if stale_dir.exists():
            shutil.rmtree(stale_dir, ignore_errors=True)
    for old in FIG_ROOT.glob("*.png"):
        old.unlink()
    for old in FIG_ROOT.glob("*.svg"):
        old.unlink()


def parse_air_file(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx: int | None = None
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if not s:
            continue
        if "wavelength" in s and ("," in s or "\t" in s):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not parse spectrum header in {path}")

    df = pd.read_csv(pd.io.common.StringIO("\n".join(lines[header_idx:])), sep=",", engine="python")
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    wl_col = next((c for c in cols if "wavelength" in c), None)
    int_col = next((c for c in cols if "irradiance" in c or "intensity" in c), None)
    if wl_col is None:
        raise ValueError(f"No wavelength column in {path}")
    if int_col is None:
        int_col = next((c for c in cols if c != wl_col), None)
    if int_col is None:
        raise ValueError(f"No intensity-like column in {path}")

    out = df[[wl_col, int_col]].copy()
    out.columns = ["wavelength_nm", "intensity"]
    out["wavelength_nm"] = pd.to_numeric(out["wavelength_nm"], errors="coerce")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
    out = out.dropna(subset=["wavelength_nm", "intensity"]).sort_values("wavelength_nm").reset_index(drop=True)
    return out


def load_air_long() -> pd.DataFrame:
    if AIR_LONG_CSV.exists():
        df = pd.read_csv(AIR_LONG_CSV)
        needed = {"sample_id", "param_set", "wavelength_nm", "irradiance_W_m2_nm"}
        if needed.issubset(df.columns):
            out = df.copy()
            out["param_set"] = out["param_set"].astype(str)
            out = out[out["param_set"].isin(AIR_LEVEL_MAP.keys())].copy()
            out = out.rename(columns={"irradiance_W_m2_nm": "intensity"})
            return out[["sample_id", "param_set", "wavelength_nm", "intensity"]].reset_index(drop=True)

    rows: List[pd.DataFrame] = []
    for path in sorted(AIR_RAW_DIR.glob("*.csv")):
        stem = path.stem
        parts = stem.split(".")
        trial = parts[-1] if len(parts) > 1 else "1"
        base = stem if stem in AIR_LEVEL_MAP else (".".join(parts[:-1]) if len(parts) > 1 else stem)
        if base not in AIR_LEVEL_MAP:
            continue
        spec = parse_air_file(path)
        spec.insert(0, "sample_id", f"air__{base}.{trial}__bulk")
        spec.insert(1, "param_set", base)
        rows.append(spec)
    if not rows:
        return pd.DataFrame(columns=["sample_id", "param_set", "wavelength_nm", "intensity"])
    return pd.concat(rows, ignore_index=True)


def baseline_correct_window(wl: np.ndarray, y: np.ndarray, start_nm: float, end_nm: float) -> float:
    mask = (wl >= start_nm) & (wl <= end_nm)
    if int(mask.sum()) < 2:
        return 0.0
    wl_win = wl[mask]
    y_win = y[mask]
    baseline = np.linspace(y_win[0], y_win[-1], y_win.size)
    corrected = np.clip(y_win - baseline, 0.0, None)
    return trapz(corrected, wl_win)


def per_spectrum_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for sample_id, g in df.groupby("sample_id", dropna=False):
        g = g.sort_values("wavelength_nm")
        wl = g["wavelength_nm"].to_numpy(dtype=float)
        y = g["intensity"].to_numpy(dtype=float)
        if wl.size < 2:
            continue
        level_key = str(g["param_set"].iloc[0])
        if level_key not in AIR_LEVEL_MAP:
            continue
        air_level = AIR_LEVEL_MAP[level_key]

        bg = float(np.nanpercentile(y, 5))
        y_bg = np.clip(y - bg, 0.0, None)
        total_area = trapz(y_bg, wl)
        if not np.isfinite(total_area) or total_area <= 0:
            total_area = float("nan")

        for species in SPECIES_ORDER:
            start_nm, end_nm = SPECIES_WINDOWS[species]
            auc_raw = baseline_correct_window(wl, y_bg, start_nm, end_nm)
            auc_norm = float(auc_raw / total_area) if np.isfinite(total_area) and total_area > 0 else float("nan")
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "air_input_level": air_level,
                    "species": species,
                    "auc_raw": auc_raw,
                    "auc_normalized": auc_norm,
                    "total_area": total_area,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["sample_id", "air_input_level", "species", "auc_raw", "auc_normalized", "total_area"]
        )
    return pd.DataFrame(rows)


def aggregate_condition_table(per_spec: pd.DataFrame) -> pd.DataFrame:
    if per_spec.empty:
        return pd.DataFrame(columns=["air_input_level", "species", "auc_raw", "auc_normalized"])

    species_auc = (
        per_spec.groupby(["air_input_level", "species"], dropna=False)[["auc_raw", "auc_normalized"]]
        .mean()
        .reset_index()
    )
    total_by_sample = per_spec[["sample_id", "air_input_level", "total_area"]].drop_duplicates(ignore_index=True)
    total_by_level = (
        total_by_sample.groupby("air_input_level", dropna=False)["total_area"].mean().rename("condition_total_area")
    )
    out = species_auc.merge(total_by_level, on="air_input_level", how="left")
    out["auc_normalized"] = out["auc_raw"] / out["condition_total_area"].replace(0, np.nan)
    out = out.drop(columns=["condition_total_area"])

    out["air_input_level"] = pd.Categorical(out["air_input_level"], categories=AIR_LEVEL_ORDER, ordered=True)
    out["species"] = pd.Categorical(out["species"], categories=SPECIES_ORDER, ordered=True)
    out = out.sort_values(["air_input_level", "species"]).reset_index(drop=True)
    out["air_input_level"] = out["air_input_level"].astype(str)
    out["species"] = out["species"].astype(str)
    return out


def save_group1_plots(df: pd.DataFrame, y_max: float, start_index: int) -> List[Path]:
    written: List[Path] = []
    levels_present = [lvl for lvl in AIR_LEVEL_ORDER if (df["air_input_level"] == lvl).any()]
    for offset, level in enumerate(levels_present):
        fig_num = start_index + offset
        d = df[df["air_input_level"] == level].copy()
        if d.empty:
            continue
        d["species"] = pd.Categorical(d["species"], categories=SPECIES_ORDER, ordered=True)
        d = d.sort_values("species")
        x = np.arange(len(SPECIES_ORDER), dtype=float)
        y = d["auc_normalized"].to_numpy(dtype=float)
        colors = [SPECIES_COLORS[s] for s in d["species"].astype(str).tolist()]

        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        ax.bar(
            x,
            y,
            color=colors,
            edgecolor="#1f1f1f",
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )
        ax.set_xlim(-0.5, len(SPECIES_ORDER) - 0.5)
        ax.set_ylim(0.0, y_max)
        ax.set_xticks(x)
        ax.set_xticklabels(SPECIES_ORDER)
        ax.set_xlabel("Species")
        ax.set_ylabel("Normalized AUC (AUC_species / AUC_total) [a.u.]")
        ax.set_title(f"Fig{fig_num}. {AIR_LEVEL_DISPLAY[level]} AUC vs Species")
        style_axes(ax, grid_axis="y")

        handles = [plt.Rectangle((0, 0), 1, 1, color=SPECIES_COLORS[sp], label=sp) for sp in SPECIES_ORDER]
        ax.legend(handles=handles, title="Species", loc="upper right", fontsize=8.8, title_fontsize=9.0)
        fig.tight_layout()

        png = FIG_ROOT / f"Fig{fig_num}.png"
        fig.savefig(png, dpi=PNG_DPI)
        plt.close(fig)
        written.append(png)
    return written


def save_group2_plots(df: pd.DataFrame, y_max: float, start_index: int) -> List[Path]:
    written: List[Path] = []
    levels_present = [lvl for lvl in AIR_LEVEL_ORDER if (df["air_input_level"] == lvl).any()]
    x = np.arange(len(levels_present), dtype=float)
    x_labels = [AIR_LEVEL_DISPLAY[level] for level in levels_present]
    for offset, species in enumerate(GROUP2_SPECIES_ORDER):
        fig_num = start_index + offset
        d = df[df["species"] == species].copy()
        if d.empty:
            continue
        d = d[d["air_input_level"].isin(levels_present)].copy()
        d["air_input_level"] = pd.Categorical(d["air_input_level"], categories=levels_present, ordered=True)
        d = d.sort_values("air_input_level")
        y = d["auc_normalized"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        ax.plot(
            x,
            y,
            color=SPECIES_COLORS[species],
            marker="o",
            markersize=6.8,
            linewidth=2.0,
            markeredgecolor="#1f1f1f",
            markeredgewidth=0.5,
            zorder=3,
        )
        ax.set_xlim(-0.2, len(levels_present) - 0.8)
        ax.set_ylim(0.0, y_max)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Air Input Level")
        ax.set_ylabel("Normalized AUC (AUC_species / AUC_total) [a.u.]")
        ax.set_title(f"Fig{fig_num}. {species} AUC vs Air Input")
        style_axes(ax, grid_axis="y")
        fig.tight_layout()

        png = FIG_ROOT / f"Fig{fig_num}.png"
        fig.savefig(png, dpi=PNG_DPI)
        plt.close(fig)
        written.append(png)
    return written


def validate_complete(table: pd.DataFrame) -> None:
    levels_present = [lvl for lvl in AIR_LEVEL_ORDER if (table["air_input_level"] == lvl).any()]
    expected = {(lvl, sp) for lvl in levels_present for sp in SPECIES_ORDER}
    got = {(str(r["air_input_level"]), str(r["species"])) for _, r in table.iterrows()}
    missing = sorted(expected - got)
    if missing:
        raise ValueError(f"Missing level/species outputs: {missing}")


def main() -> int:
    apply_publication_style()
    ensure_all_scope_layouts()
    np.random.seed(0)

    df = load_air_long()
    if df.empty:
        print("No usable air spectra found.")
        return 1

    per_spec = per_spectrum_auc(df)
    if per_spec.empty:
        print("No per-spectrum AUC rows produced.")
        return 2

    table = aggregate_condition_table(per_spec)
    validate_complete(table)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    prepare_output_dirs()
    table.to_csv(CSV_OUT, index=False, columns=["air_input_level", "species", "auc_raw", "auc_normalized"])

    ymax_data = float(np.nanmax(table["auc_normalized"].to_numpy(dtype=float))) if not table.empty else 0.0
    y_max = max(1e-6, ymax_data * 1.12)

    written_figs: List[Path] = []
    group1 = save_group1_plots(table, y_max=y_max, start_index=1)
    group2 = save_group2_plots(table, y_max=y_max, start_index=1 + len(group1))
    written_figs.extend(group1)
    written_figs.extend(group2)

    print(f"Wrote {CSV_OUT}")
    for path in written_figs:
        print(f"Wrote {path}")
    print(
        f"Done. rows={len(table)} figures={len(written_figs)} "
        f"figure_dir={FIG_ROOT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
