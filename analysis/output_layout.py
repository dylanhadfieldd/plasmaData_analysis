#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from analysis.plot_style import apply_publication_style, spectral_interval_label, style_axes, to_species_label
except ModuleNotFoundError:
    from plot_style import apply_publication_style, spectral_interval_label, style_axes, to_species_label

ROOT_OUTPUT = Path("output")
STATS_ROOT = Path("stats")
MS_SRC = Path("msOutput")
CHEM_SRC = Path("chemSpecies_output")
TMP_ROOT = Path("_output_relayout_tmp")

SCOPES = ("air", "diameter", "meta")
SCOPE_DIRS = ("spectral", "chemspecies", "pca")
LABELED_TOP_N = 8

MS_RAW_FILES = (
    "averaged_curves_long.csv",
    "averaged_peaks_top10.csv",
    "trial_peaks_top10.csv",
    "nist_matches_top3.csv",
    "target_species_peak_matches.csv",
    "target_species_match_summary.csv",
    "nist_fetch_status.csv",
)

CHEM_CSV_KEEP = (
    "group_species_concentration_long.csv",
    "group_species_concentration_wide.csv",
    "dataset_species_concentration_summary.csv",
    "air_vs_diameter_species_delta.csv",
    "key_species_group_findings.csv",
)

CHEM_FIG_KEEP = {
    "air": (
        "air_group_species_concentration_heatmap.png",
        "air_species_concentration_mix.png",
        "air_species_rank_with_variability.png",
        "air_param_species_heatmap.png",
    ),
    "diameter": (
        "diameter_group_species_concentration_heatmap.png",
        "diameter_species_concentration_mix.png",
        "diameter_species_rank_with_variability.png",
        "diameter_param_species_heatmap.png",
    ),
    "meta": (
        "fig1_group_species_concentration_heatmap.png",
        "fig2_dataset_species_concentration_mix.png",
        "fig3_air_vs_diameter_species_delta.png",
        "fig4_meta_group_total_signal.png",
    ),
}


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copy_any(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def safe_name(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip())
    out = "_".join([p for p in out.split("_") if p])
    return out or "group"


def ensure_dirs(root: Path) -> None:
    for scope in SCOPES:
        (root / scope / "spectral" / "base" / "raw").mkdir(parents=True, exist_ok=True)
        (root / scope / "spectral" / "base" / "charts" / "individual").mkdir(parents=True, exist_ok=True)
        (root / scope / "spectral" / "base" / "charts" / "composed").mkdir(parents=True, exist_ok=True)
        (root / scope / "spectral" / "base" / "charts" / "compared").mkdir(parents=True, exist_ok=True)
        (root / scope / "spectral" / "labels").mkdir(parents=True, exist_ok=True)
        (root / scope / "chemspecies" / "csv").mkdir(parents=True, exist_ok=True)
        (root / scope / "chemspecies" / "figures").mkdir(parents=True, exist_ok=True)
        (root / scope / "pca").mkdir(parents=True, exist_ok=True)


def copy_base_spectral(scope: str, tmp_root: Path) -> None:
    base_root = tmp_root / scope / "spectral" / "base"
    raw_dir = base_root / "raw"
    charts_dir = base_root / "charts"

    if scope in {"air", "diameter"}:
        for name in ("metadata.csv", "spectra_long.csv", "spectra_wide.csv"):
            copy_any(ROOT_OUTPUT / scope / name, raw_dir / name)
        copy_any(ROOT_OUTPUT / "charts" / scope, charts_dir / "individual")
        copy_any(ROOT_OUTPUT / "composed" / scope, charts_dir / "composed")
        copy_any(ROOT_OUTPUT / "combined" / scope, charts_dir / "compared")
    else:
        for name in ("metadata.csv", "spectra_long.csv", "spectra_wide.csv"):
            copy_any(ROOT_OUTPUT / name, raw_dir / name)
        for name in ("combined.png", "combined_averages_long.csv"):
            copy_any(ROOT_OUTPUT / "combined" / name, charts_dir / "compared" / name)


def copy_pca(scope: str, tmp_root: Path) -> None:
    src_scope = scope if scope in {"air", "diameter"} else "combined"
    copy_any(STATS_ROOT / "pca" / src_scope, tmp_root / scope / "pca")


def split_scope_rows(df: pd.DataFrame, scope: str, allow_global: bool) -> pd.DataFrame:
    if scope == "meta":
        return df.copy()
    if "dataset" not in df.columns:
        return df.copy() if allow_global else pd.DataFrame(columns=df.columns)
    d = df[df["dataset"].astype(str).str.lower() == scope].copy()
    return d


def copy_ms_raw(tmp_root: Path) -> None:
    for name in MS_RAW_FILES:
        src = MS_SRC / name
        if not src.exists():
            continue
        try:
            df = pd.read_csv(src)
        except Exception:
            continue
        for scope in SCOPES:
            allow_global = name in {"nist_fetch_status.csv"}
            part = split_scope_rows(df, scope, allow_global=allow_global)
            if part.empty:
                continue
            out = tmp_root / scope / "spectral" / "base" / "raw" / name
            part.to_csv(out, index=False)


def build_labeled_traceability(scope_dir: Path) -> pd.DataFrame:
    target_csv = scope_dir / "spectral" / "base" / "raw" / "target_species_peak_matches.csv"
    peaks_csv = scope_dir / "spectral" / "base" / "raw" / "averaged_peaks_top10.csv"
    nist_csv = scope_dir / "spectral" / "base" / "raw" / "nist_matches_top3.csv"
    fetch_csv = scope_dir / "spectral" / "base" / "raw" / "nist_fetch_status.csv"

    frames = []

    if not target_csv.exists():
        target = pd.DataFrame()
    else:
        target = pd.read_csv(target_csv)

    if not target.empty:
        if "matched" in target.columns:
            target = target[target["matched"].astype(bool)].copy()
        if not target.empty:
            t = pd.DataFrame(
                {
                    "dataset": target.get("dataset", np.nan),
                    "param_set": target.get("param_set", np.nan),
                    "channel": target.get("channel", np.nan),
                    "peak_rank": target.get("matched_peak_rank", np.nan),
                    "peak_wavelength_nm_0p1": target.get("matched_peak_wavelength_nm_0p1", np.nan),
                    "peak_intensity_refined": target.get("matched_peak_intensity", np.nan),
                    "delta_nm": target.get("delta_nm", np.nan),
                    "trace_species": target.get("species", "").fillna("").astype(str),
                    "target_wavelength_nm": target.get("target_wavelength_nm", np.nan),
                    "source_url": "configs/target_species_lines.csv",
                    "status": "target_match",
                }
            )
            frames.append(t)

    if nist_csv.exists():
        nist = pd.read_csv(nist_csv)
        if not nist.empty:
            nist = nist[pd.to_numeric(nist.get("candidate_rank"), errors="coerce") == 1].copy()
            nist["peak_wavelength_nm_0p1"] = pd.to_numeric(nist.get("peak_wavelength_nm_0p1"), errors="coerce")
            nist = nist[nist["peak_wavelength_nm_0p1"] < 300.0].copy()
            if not nist.empty:
                if "nist_species" in nist.columns:
                    nist["trace_species"] = nist["nist_species"].fillna("").astype(str)
                elif "nist_spectra_query" in nist.columns:
                    nist["trace_species"] = nist["nist_spectra_query"].fillna("").astype(str)
                else:
                    nist["trace_species"] = "unknown"
            if not nist.empty:

                nist["source_url"] = ""
                nist["status"] = "nist_sub300_candidate"
                if fetch_csv.exists():
                    fetch = pd.read_csv(fetch_csv)
                    if {"spectra", "source_url"}.issubset(fetch.columns) and "nist_spectra_query" in nist.columns:
                        join = fetch.rename(columns={"spectra": "nist_spectra_query"})
                        nist = nist.merge(
                            join[["nist_spectra_query", "source_url"]],
                            on="nist_spectra_query",
                            how="left",
                            suffixes=("", "_join"),
                        )
                        nist["source_url"] = nist["source_url_join"].fillna(nist["source_url"])
                        nist = nist.drop(columns=["source_url_join"], errors="ignore")

                sub = pd.DataFrame(
                    {
                        "dataset": nist.get("dataset", np.nan),
                        "param_set": nist.get("param_set", np.nan),
                        "channel": nist.get("channel", np.nan),
                        "peak_rank": nist.get("peak_rank", np.nan),
                        "peak_wavelength_nm_0p1": nist.get("peak_wavelength_nm_0p1", np.nan),
                        "peak_intensity_refined": nist.get("peak_intensity_refined", nist.get("peak_intensity", np.nan)),
                        "delta_nm": nist.get("delta_nm", np.nan),
                        "trace_species": nist.get("trace_species", "").fillna("").astype(str),
                        "target_wavelength_nm": np.nan,
                        "source_url": nist.get("source_url", ""),
                        "status": nist.get("status", "nist_sub300_candidate"),
                    }
                )
                frames.append(sub)

    if peaks_csv.exists():
        peaks = pd.read_csv(peaks_csv)
        if not peaks.empty:
            peaks["peak_wavelength_nm_0p1"] = pd.to_numeric(peaks.get("peak_wavelength_nm_0p1"), errors="coerce")
            peaks = peaks[peaks["peak_wavelength_nm_0p1"] < 300.0].copy()
            if not peaks.empty:
                has_nist = pd.DataFrame()
                if frames:
                    all_frames = pd.concat(frames, ignore_index=True)
                    has_nist = all_frames[all_frames.get("status", "") == "nist_sub300_candidate"].copy()
                if not has_nist.empty:
                    nist_keys = (
                        has_nist[["dataset", "param_set", "channel", "peak_wavelength_nm_0p1"]]
                        .astype(str)
                        .drop_duplicates()
                    )
                    peaks_keys = peaks[["dataset", "param_set", "channel", "peak_wavelength_nm_0p1"]].astype(str)
                    missing_mask = ~peaks_keys.apply(tuple, axis=1).isin(nist_keys.apply(tuple, axis=1))
                    missing = peaks[missing_mask].copy()
                else:
                    missing = peaks.copy()

                if not missing.empty:
                    fallback = pd.DataFrame(
                        {
                            "dataset": missing.get("dataset", np.nan),
                            "param_set": missing.get("param_set", np.nan),
                            "channel": missing.get("channel", np.nan),
                            "peak_rank": missing.get("peak_rank", np.nan),
                            "peak_wavelength_nm_0p1": missing.get("peak_wavelength_nm_0p1", np.nan),
                            "peak_intensity_refined": missing.get("peak_intensity_refined", missing.get("peak_intensity", np.nan)),
                            "delta_nm": np.nan,
                            "trace_species": "Unassigned",
                            "target_wavelength_nm": np.nan,
                            "source_url": "",
                            "status": "sub300_unassigned",
                        }
                    )
                    frames.append(fallback)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["trace_species"] = out["trace_species"].astype(str).str.strip()
    out = out[out["trace_species"] != ""].copy()
    out["peak_wavelength_nm_0p1"] = pd.to_numeric(out["peak_wavelength_nm_0p1"], errors="coerce")
    out = out.dropna(subset=["peak_wavelength_nm_0p1"]).copy()
    out = out.drop_duplicates(
        subset=["dataset", "param_set", "channel", "peak_wavelength_nm_0p1", "trace_species"],
        keep="first",
    )
    return out.sort_values(["param_set", "channel", "peak_wavelength_nm_0p1", "trace_species"], ignore_index=True)


def annotate_group_chart(ax: plt.Axes, label_df: pd.DataFrame) -> None:
    top = label_df.copy()
    top["peak_intensity_refined"] = pd.to_numeric(top["peak_intensity_refined"], errors="coerce")
    top["peak_wavelength_nm_0p1"] = pd.to_numeric(top["peak_wavelength_nm_0p1"], errors="coerce")
    top["target_wavelength_nm"] = pd.to_numeric(top.get("target_wavelength_nm"), errors="coerce")
    top["label_priority"] = top.get("status", "").map(
        {
            "target_match": 0,
            "nist_sub300_candidate": 1,
            "nist_sub300_oxygen_candidate": 1,
            "sub300_unassigned": 2,
        }
    ).fillna(3)
    top = top[np.isfinite(top["peak_intensity_refined"]) & np.isfinite(top["peak_wavelength_nm_0p1"])].copy()
    if top.empty:
        return

    targeted = top[np.isfinite(top["target_wavelength_nm"])].copy()
    targeted = targeted.sort_values(["target_wavelength_nm", "peak_intensity_refined"], ascending=[True, False])
    sub300 = top[top["peak_wavelength_nm_0p1"] < 300.0].copy()
    sub300 = sub300.sort_values(
        ["peak_wavelength_nm_0p1", "label_priority", "peak_intensity_refined"],
        ascending=[True, True, False],
    )

    merged = pd.concat([targeted, sub300], ignore_index=True)
    merged = merged.sort_values(
        ["peak_wavelength_nm_0p1", "label_priority", "peak_intensity_refined"],
        ascending=[True, True, False],
        ignore_index=True,
    )
    merged = merged.drop_duplicates(subset=["peak_wavelength_nm_0p1"], keep="first")
    if len(merged) > max(LABELED_TOP_N, 12):
        merged = merged.head(max(LABELED_TOP_N, 12))

    for _, row in merged.iterrows():
        x = float(row["peak_wavelength_nm_0p1"])
        y = float(row["peak_intensity_refined"])
        species = to_species_label(str(row["trace_species"]))
        target_wl = row["target_wavelength_nm"]
        if np.isfinite(target_wl):
            label = f"{species} [{float(target_wl):.0f}, {x:.1f} nm]"
        elif str(row.get("status", "")) in {"nist_sub300_candidate", "nist_sub300_oxygen_candidate"}:
            label = f"{species} @ {x:.1f} nm (NIST)"
        else:
            label = f"Unassigned sub300 @ {x:.1f} nm"
        ax.scatter([x], [y], color="#d1495b", s=26, zorder=4)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
            color="#222222",
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#666666"},
        )


def write_labeled_assets(scope: str, scope_dir: Path) -> None:
    curves_csv = scope_dir / "spectral" / "base" / "raw" / "averaged_curves_long.csv"
    labels_dir = scope_dir / "spectral" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    trace = build_labeled_traceability(scope_dir)
    if not trace.empty:
        trace.to_csv(labels_dir / "labeled_peak_traceability.csv", index=False)
    if not curves_csv.exists() or trace.empty:
        return

    curves = pd.read_csv(curves_csv)
    if curves.empty:
        return

    group_cols: Sequence[str] = ("param_set", "channel")
    if scope == "meta" and "dataset" in curves.columns:
        group_cols = ("dataset", "param_set", "channel")

    for key, g in curves.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        filt = np.ones(len(trace), dtype=bool)
        for col, val in zip(group_cols, key):
            filt &= trace[col].astype(str) == str(val)
        t = trace[filt].copy()
        if t.empty:
            continue

        g = g.sort_values("wavelength_nm")
        y_raw = g["irradiance_mean"].to_numpy(dtype=float)
        scale = float(np.nanmax(y_raw)) if np.isfinite(y_raw).any() else 0.0
        if scale > 0:
            y_plot = y_raw / scale
        else:
            y_plot = np.zeros_like(y_raw, dtype=float)
        t = t.copy()
        t["peak_intensity_refined"] = pd.to_numeric(t["peak_intensity_refined"], errors="coerce")
        if scale > 0:
            t["peak_intensity_refined"] = t["peak_intensity_refined"] / scale
        else:
            t["peak_intensity_refined"] = 0.0

        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        ax.plot(
            g["wavelength_nm"].to_numpy(dtype=float),
            y_plot,
            color="#1f4e79",
            linewidth=1.8,
            alpha=0.95,
        )
        annotate_group_chart(ax, t)
        ax.set_title(f"{scope} Labeled Spectrum | " + " | ".join(str(v) for v in key))
        ax.set_xlabel(f"Wavelength {spectral_interval_label(g['wavelength_nm'].to_numpy(dtype=float))}")
        ax.set_ylabel("Normalized Irradiance (a.u.)")
        ax.set_ylim(0, 1.05)
        style_axes(ax, grid_axis="both")
        out_name = "_".join(safe_name(v) for v in key) + "_labeled.png"
        fig.tight_layout()
        fig.savefig(labels_dir / out_name, dpi=220)
        plt.close(fig)

    note = scope_dir / "spectral" / "base" / "raw" / "source_notes.txt"
    note.write_text(
        "NIST source: https://www.nist.gov/pml/atomic-spectra-database\n"
        "ASD query endpoint: https://physics.nist.gov/cgi-bin/ASD/lines1.pl\n"
        "Target species list: configs/target_species_lines.csv\n",
        encoding="utf-8",
    )


def copy_chemspecies(scope: str, tmp_root: Path) -> None:
    dst = tmp_root / scope / "chemspecies"
    src_csv = CHEM_SRC / "csv"
    src_fig = CHEM_SRC / "figures"

    if src_csv.exists():
        for name in CHEM_CSV_KEEP:
            path = src_csv / name
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if scope == "meta":
                out = df
            elif "dataset" in df.columns:
                out = df[df["dataset"].astype(str).str.lower() == scope].copy()
            elif name == "air_vs_diameter_species_delta.csv":
                out = df
            else:
                out = pd.DataFrame(columns=df.columns)
            if out.empty:
                continue
            out.to_csv(dst / "csv" / name, index=False)

    if src_fig.exists():
        for fig_name in CHEM_FIG_KEEP[scope]:
            copy_any(src_fig / fig_name, dst / "figures" / fig_name)


def finalize_structure(tmp_root: Path) -> None:
    ROOT_OUTPUT.mkdir(parents=True, exist_ok=True)
    for scope in SCOPES:
        src_scope = tmp_root / scope
        dst_scope = ROOT_OUTPUT / scope
        dst_scope.mkdir(parents=True, exist_ok=True)
        for name in SCOPE_DIRS:
            src = src_scope / name
            dst = dst_scope / name
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            if src.exists():
                shutil.move(str(src), str(dst))


def cleanup_scope_noise() -> None:
    for scope in SCOPES:
        scope_dir = ROOT_OUTPUT / scope
        if not scope_dir.exists():
            continue
        for item in scope_dir.iterdir():
            keep_dir = item.is_dir() and item.name in SCOPE_DIRS
            keep_exec = item.is_file() and item.suffix.lower() == ".xlsx" and item.name.startswith(f"{scope}_executive_report")
            keep_temp_lock = item.is_file() and item.name.startswith("~$")
            if keep_dir or keep_exec or keep_temp_lock:
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try:
                    item.unlink()
                except OSError:
                    pass


def cleanup_root_noise() -> None:
    if not ROOT_OUTPUT.exists():
        return
    for item in ROOT_OUTPUT.iterdir():
        if item.is_dir() and item.name in SCOPES:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try:
                item.unlink()
            except OSError:
                pass


def remove_external_artifacts() -> None:
    if MS_SRC.exists():
        shutil.rmtree(MS_SRC, ignore_errors=True)
    if CHEM_SRC.exists():
        shutil.rmtree(CHEM_SRC, ignore_errors=True)
    if STATS_ROOT.exists():
        shutil.rmtree(STATS_ROOT, ignore_errors=True)


def main() -> int:
    apply_publication_style()
    clean_dir(TMP_ROOT)
    ensure_dirs(TMP_ROOT)

    for scope in SCOPES:
        copy_base_spectral(scope, TMP_ROOT)
        copy_pca(scope, TMP_ROOT)
        copy_chemspecies(scope, TMP_ROOT)

    copy_ms_raw(TMP_ROOT)
    for scope in SCOPES:
        write_labeled_assets(scope, TMP_ROOT / scope)

    finalize_structure(TMP_ROOT)
    cleanup_scope_noise()
    cleanup_root_noise()
    clean_dir(TMP_ROOT)
    remove_external_artifacts()

    print("Output relayout complete:")
    print("  output/*/spectral/base")
    print("  output/*/spectral/labels")
    print("  output/*/chemspecies")
    print("  output/*/pca")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
