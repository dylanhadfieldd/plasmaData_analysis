from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.output_paths import metadata_csv_path, spectral_labels_dir
from analysis.plot_style import spectral_interval_label, style_axes, to_species_label

LABELED_TOP_N = 8
REACTIVE_NIST_SPECIES = {"N", "O", "HE"}


def nist_species_in_scope(value: object) -> bool:
    text = str(value).strip()
    if not text:
        return False
    token = text.split()[0].upper()
    token = "".join(ch for ch in token if ch.isalpha())
    return token in REACTIVE_NIST_SPECIES


def build_labeled_traceability(scope: str) -> pd.DataFrame:
    target_csv = metadata_csv_path(scope, "spectral", "target_species_peak_matches.csv")
    peaks_csv = metadata_csv_path(scope, "spectral", "averaged_peaks_top10.csv")
    nist_csv = metadata_csv_path(scope, "spectral", "nist_matches_top3.csv")
    fetch_csv = metadata_csv_path(scope, "spectral", "nist_fetch_status.csv")

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
                    "candidate_rank": np.nan,
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
            nist["candidate_rank"] = pd.to_numeric(nist.get("candidate_rank"), errors="coerce")
            nist["peak_wavelength_nm_0p1"] = pd.to_numeric(nist.get("peak_wavelength_nm_0p1"), errors="coerce")
            nist = nist[np.isfinite(nist["peak_wavelength_nm_0p1"])].copy()
            if not nist.empty:
                if "nist_species" in nist.columns:
                    nist["trace_species"] = nist["nist_species"].fillna("").astype(str)
                elif "nist_spectra_query" in nist.columns:
                    nist["trace_species"] = nist["nist_spectra_query"].fillna("").astype(str)
                else:
                    nist["trace_species"] = "unknown"
                nist = nist[nist["trace_species"].map(nist_species_in_scope)].copy()
            if not nist.empty:
                nist = nist.sort_values(
                    ["dataset", "param_set", "channel", "peak_rank", "candidate_rank"],
                    ascending=[True, True, True, True, True],
                    ignore_index=True,
                )

                nist["source_url"] = ""
                nist["status"] = "nist_reactive_candidate"
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
                        "candidate_rank": nist.get("candidate_rank", np.nan),
                        "peak_rank": nist.get("peak_rank", np.nan),
                        "peak_wavelength_nm_0p1": nist.get("peak_wavelength_nm_0p1", np.nan),
                        "peak_intensity_refined": nist.get("peak_intensity_refined", nist.get("peak_intensity", np.nan)),
                        "delta_nm": nist.get("delta_nm", np.nan),
                        "trace_species": nist.get("trace_species", "").fillna("").astype(str),
                        "target_wavelength_nm": np.nan,
                        "source_url": nist.get("source_url", ""),
                        "status": nist.get("status", "nist_reactive_candidate"),
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
                    has_nist = all_frames[
                        all_frames.get("status", "").isin({"nist_reactive_candidate", "nist_sub300_candidate"})
                    ].copy()
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
                            "candidate_rank": np.nan,
                            "peak_rank": missing.get("peak_rank", np.nan),
                            "peak_wavelength_nm_0p1": missing.get("peak_wavelength_nm_0p1", np.nan),
                            "peak_intensity_refined": missing.get(
                                "peak_intensity_refined", missing.get("peak_intensity", np.nan)
                            ),
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
        subset=["dataset", "param_set", "channel", "peak_wavelength_nm_0p1", "trace_species", "status"],
        keep="first",
    )
    return out.sort_values(["param_set", "channel", "peak_wavelength_nm_0p1", "trace_species"], ignore_index=True)


def annotate_group_chart(ax: plt.Axes, label_df: pd.DataFrame) -> None:
    top = label_df.copy()
    top["candidate_rank"] = pd.to_numeric(top.get("candidate_rank"), errors="coerce")
    top["peak_intensity_refined"] = pd.to_numeric(top["peak_intensity_refined"], errors="coerce")
    top["peak_wavelength_nm_0p1"] = pd.to_numeric(top["peak_wavelength_nm_0p1"], errors="coerce")
    top["target_wavelength_nm"] = pd.to_numeric(top.get("target_wavelength_nm"), errors="coerce")
    top["label_priority"] = top.get("status", "").map(
        {
            "target_match": 0,
            "nist_reactive_candidate": 1,
            "nist_sub300_candidate": 1,
            "nist_sub300_oxygen_candidate": 1,
            "sub300_unassigned": 2,
        }
    ).fillna(3)
    top["candidate_rank_priority"] = top["candidate_rank"].where(np.isfinite(top["candidate_rank"]), 999.0)
    top = top[np.isfinite(top["peak_intensity_refined"]) & np.isfinite(top["peak_wavelength_nm_0p1"])].copy()
    if top.empty:
        return

    targeted = top[np.isfinite(top["target_wavelength_nm"])].copy()
    targeted = targeted.sort_values(["target_wavelength_nm", "peak_intensity_refined"], ascending=[True, False])
    candidates = top.copy()
    candidates = candidates.sort_values(
        ["peak_wavelength_nm_0p1", "label_priority", "candidate_rank_priority", "peak_intensity_refined"],
        ascending=[True, True, True, False],
    )

    merged = pd.concat([targeted, candidates], ignore_index=True)
    merged = merged.sort_values(
        ["peak_wavelength_nm_0p1", "label_priority", "candidate_rank_priority", "peak_intensity_refined"],
        ascending=[True, True, True, False],
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
        elif str(row.get("status", "")) in {
            "nist_reactive_candidate",
            "nist_sub300_candidate",
            "nist_sub300_oxygen_candidate",
        }:
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


def write_labeled_assets(scope: str) -> None:
    curves_csv = metadata_csv_path(scope, "spectral", "averaged_curves_long.csv")
    trace_csv = metadata_csv_path(scope, "spectral", "labeled_peak_traceability.csv")
    labels_dir = spectral_labels_dir(scope)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for old in labels_dir.glob("*.png"):
        old.unlink()

    trace = build_labeled_traceability(scope)
    if not trace.empty:
        trace_csv.parent.mkdir(parents=True, exist_ok=True)
        trace.to_csv(trace_csv, index=False)
    if not curves_csv.exists() or trace.empty:
        return

    curves = pd.read_csv(curves_csv)
    if curves.empty:
        return

    group_cols: Sequence[str] = ("param_set", "channel")
    if scope == "meta" and "dataset" in curves.columns:
        group_cols = ("dataset", "param_set", "channel")

    for index, (key, g) in enumerate(curves.groupby(list(group_cols), dropna=False), start=1):
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
        out_name = f"Fig{index}.png"
        fig.tight_layout()
        labels_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(labels_dir / out_name, dpi=220)
        plt.close(fig)

    note = metadata_csv_path(scope, "spectral", "source_notes.txt")
    note.write_text(
        "NIST source: https://www.nist.gov/pml/atomic-spectra-database\n"
        "ASD query endpoint: https://physics.nist.gov/cgi-bin/ASD/lines1.pl\n"
        "Target species list: configs/target_species_lines.csv\n",
        encoding="utf-8",
    )
