#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

try:
    from analysis.plot_style import apply_publication_style, style_axes, to_species_label
except ModuleNotFoundError:
    from plot_style import apply_publication_style, style_axes, to_species_label

try:
    from openpyxl.drawing.image import Image as XLImage
except Exception:  # pragma: no cover
    XLImage = None

OUTPUT_ROOT = Path("output")
SCOPES = ("air", "diameter", "meta")
MAX_SUMMARY_FIGURES = 4
PREFERRED_PARAM_ORDER = {
    "air": ["100H", "5H..9A", "5H..5A", "5H..01A"],
    "diameter": ["1mm", "0.5mm"],
}


def read_csv_or_note(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame([{"note": f"Missing file: {path.as_posix()}"}])
    try:
        return pd.read_csv(path)
    except Exception as e:
        return pd.DataFrame([{"note": f"Could not read {path.as_posix()}: {e}"}])


def write_dataframe(ws, df: pd.DataFrame, start_row: int = 1, start_col: int = 1) -> int:
    if df.empty:
        ws.cell(row=start_row, column=start_col, value="(no rows)")
        return start_row + 1

    cols = [str(c) for c in df.columns]
    for j, col in enumerate(cols, start=start_col):
        cell = ws.cell(row=start_row, column=j, value=col)
        cell.font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    r = start_row + 1
    for row in df.itertuples(index=False, name=None):
        for j, val in enumerate(row, start=start_col):
            ws.cell(row=r, column=j, value=None if (isinstance(val, float) and np.isnan(val)) else val)
        r += 1
    return r


def sanitize_sheet_name(name: str, used_names: set[str]) -> str:
    invalid = set(r'[]:*?/\\')
    clean = "".join(ch for ch in str(name) if ch not in invalid).strip()
    clean = clean or "Trials"
    clean = clean[:31]
    if clean not in used_names:
        return clean
    base = clean[:28]
    idx = 2
    while True:
        candidate = f"{base}_{idx}"[:31]
        if candidate not in used_names:
            return candidate
        idx += 1


def safe_name(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip())
    out = "_".join([p for p in out.split("_") if p])
    return out or "group"


def ordered_param_sets(scope: str, raw_long: pd.DataFrame) -> List[str]:
    if "param_set" not in raw_long.columns:
        return []
    vals = raw_long["param_set"].astype(str).dropna().unique().tolist()
    preferred = PREFERRED_PARAM_ORDER.get(scope, [])
    ordered: List[str] = [p for p in preferred if p in vals]
    ordered.extend(sorted([v for v in vals if v not in ordered]))
    return ordered


def embed_image(
    ws,
    image_path: Path,
    anchor: str,
    max_w: float,
    max_h: float,
    missing_cell: str,
    missing_label: str,
) -> None:
    if XLImage is None:
        ws[missing_cell] = f"Image embedding unavailable: {missing_label}"
        return
    if not image_path.exists():
        ws[missing_cell] = f"Missing image: {missing_label}"
        return
    try:
        img = XLImage(str(image_path))
        w = float(getattr(img, "width", 1.0))
        h = float(getattr(img, "height", 1.0))
        if w > 0 and h > 0:
            scale = min(max_w / w, max_h / h, 1.0)
            img.width = int(w * scale)
            img.height = int(h * scale)
        ws.add_image(img, anchor)
    except Exception as e:
        ws[missing_cell] = f"Could not embed {missing_label}: {e}"


def build_param_concentration_figure(
    scope_dir: Path, scope: str, param: str, target_matches: pd.DataFrame
) -> Path | None:
    apply_publication_style()
    if target_matches.empty or "param_set" not in target_matches.columns:
        return None
    d = target_matches[target_matches["param_set"].astype(str) == str(param)].copy()
    if d.empty:
        return None
    if "matched" in d.columns:
        d = d[d["matched"].astype(bool)].copy()
    if d.empty:
        return None

    d["delta_nm"] = pd.to_numeric(d.get("delta_nm"), errors="coerce").fillna(np.nan)
    d["intensity"] = pd.to_numeric(d.get("matched_peak_intensity"), errors="coerce").fillna(0.0)
    d["score"] = d["intensity"] * (1.0 / (1.0 + d["delta_nm"].abs().fillna(0.0)))
    d["species"] = d.get("species", "").fillna("").astype(str)
    d = d[d["species"] != ""]
    if d.empty:
        return None

    if "channel" in d.columns:
        pivot = d.pivot_table(index="species", columns="channel", values="score", aggfunc="sum", fill_value=0.0)
    else:
        pivot = d.groupby("species", dropna=False)["score"].sum().to_frame("all")
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    species_labels = [to_species_label(v) for v in pivot.index.astype(str).tolist()]
    if pivot.shape[1] == 1:
        vals = pivot.iloc[:, 0].to_numpy(dtype=float)
        ax.bar(species_labels, vals, color="#2a9d8f", alpha=0.92, edgecolor="#2f2f2f", linewidth=0.35)
    else:
        x = np.arange(len(pivot.index))
        width = 0.75 / max(1, pivot.shape[1])
        for i, col in enumerate(pivot.columns):
            vals = pivot[col].to_numpy(dtype=float)
            offset = (i - (pivot.shape[1] - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, alpha=0.92, label=str(col), edgecolor="#2f2f2f", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(species_labels)
        ax.legend(loc="best", fontsize=7)

    ax.set_title(f"{scope} {param} Target-Species Concentration")
    ax.set_xlabel("Species")
    ax.set_ylabel("Relative concentration score")
    ax.tick_params(axis="x", labelrotation=20, labelsize=10.8)
    style_axes(ax, grid_axis="y")
    fig.tight_layout()

    out_dir = scope_dir / "chemspecies" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(param)}_trial_concentration.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def select_key_figures(scope_dir: Path, scope: str) -> List[Tuple[str, Path]]:
    wanted = [
        ("Figure 1: Species-labeled spectrum", sorted(scope_dir.glob("spectral/labels/*_labeled.png"))),
        ("Figure 2: Unlabeled spectral comparison", sorted(scope_dir.glob("spectral/base/charts/compared/*.png"))),
        ("Figure 3: PCA score map", sorted(scope_dir.glob("pca/pca_scores.png"))),
    ]

    chem_specific = (
        f"chemspecies/figures/{scope}_species_concentration_mix.png"
        if scope in {"air", "diameter"}
        else "chemspecies/figures/fig2_dataset_species_concentration_mix.png"
    )
    wanted.append(("Figure 4: Species concentration mix", sorted(scope_dir.glob(chem_specific))))

    chosen: List[Tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, candidates in wanted:
        for c in candidates:
            if c.exists() and c not in seen:
                chosen.append((label, c))
                seen.add(c)
                break

    if len(chosen) < MAX_SUMMARY_FIGURES:
        for c in sorted(scope_dir.glob("**/*.png")):
            if c in seen:
                continue
            chosen.append((f"Figure {len(chosen)+1}: Additional key view", c))
            seen.add(c)
            if len(chosen) >= MAX_SUMMARY_FIGURES:
                break
    return chosen[:MAX_SUMMARY_FIGURES]


def parse_metric_counts(
    metadata: pd.DataFrame, avg_peaks: pd.DataFrame, target_matches: pd.DataFrame, nist_status: pd.DataFrame
) -> Tuple[int, int, int, int, int, int, float]:
    sample_count = int(metadata["sample_id"].nunique()) if "sample_id" in metadata.columns else 0
    group_count = (
        int(metadata.groupby(["param_set", "channel"], dropna=False).ngroups)
        if {"param_set", "channel"}.issubset(metadata.columns)
        else 0
    )
    avg_peak_count = int(len(avg_peaks))
    d = target_matches.copy()
    if not d.empty and "matched" in d.columns:
        d = d[d["matched"].astype(bool)]
    matched = int(len(d)) if not d.empty else 0
    targets_total = int(len(target_matches)) if not target_matches.empty else 0
    status = nist_status["status"].astype(str).str.lower() if "status" in nist_status.columns else pd.Series([], dtype=str)
    ok_queries = int((status == "ok").sum()) if not status.empty else 0
    fail_queries = int((status != "ok").sum()) if not status.empty else 0
    if targets_total > 0:
        coverage = matched / targets_total
    else:
        coverage = (matched / avg_peak_count) if avg_peak_count > 0 else 0.0
    return sample_count, group_count, avg_peak_count, matched, ok_queries, fail_queries, coverage


def top_species_text(scope: str, chem_summary: pd.DataFrame, top_n: int = 3) -> str:
    if chem_summary.empty or "species" not in chem_summary.columns:
        return "not available"
    target = scope if scope in {"air", "diameter"} else "combined"
    d = chem_summary.copy()
    if "dataset" in d.columns:
        d = d[d["dataset"].astype(str).str.lower() == target.lower()]
    if d.empty or "mean_relative_concentration" not in d.columns:
        return "not available"
    d["mean_relative_concentration"] = pd.to_numeric(d["mean_relative_concentration"], errors="coerce")
    d = d.dropna(subset=["mean_relative_concentration"])
    if d.empty:
        return "not available"
    top = d.nlargest(top_n, "mean_relative_concentration")
    return ", ".join(
        f"{str(r['species'])} ({100.0 * float(r['mean_relative_concentration']):.1f}%)"
        for _, r in top.iterrows()
    )


def pca_text(pca_var: pd.DataFrame) -> str:
    if pca_var.empty or "explained_variance_ratio" not in pca_var.columns:
        return "PCA summary unavailable."
    vals = pd.to_numeric(pca_var["explained_variance_ratio"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return "PCA summary unavailable."
    pc1 = 100.0 * float(vals[0])
    pc2 = 100.0 * float(vals[1]) if vals.size > 1 else 0.0
    if vals.size > 1:
        return f"PCA separation is concentrated in Figure 3 (PC1 {pc1:.1f}%, PC2 {pc2:.1f}%)."
    return f"PCA separation is concentrated in Figure 3 (PC1 {pc1:.1f}%)."


def narrative_lines(
    scope: str,
    sample_count: int,
    group_count: int,
    avg_peak_count: int,
    matched: int,
    coverage: float,
    top_species: str,
    pca_sentence: str,
) -> List[str]:
    return [
        f"This {scope} workbook summarizes the study with four key visuals below (Figures 1-4).",
        f"Figure 1 shows representative labeled emission structure, while Figure 2 shows unlabeled aggregate spectral comparison for the same scope.",
        f"The scope includes {sample_count} trials across {group_count} condition groups, with {avg_peak_count} averaged peaks and {matched} matched peaks ({100.0 * coverage:.1f}% coverage).",
        pca_sentence,
        f"Figure 4 highlights dominant chemical species concentrations: {top_species}.",
    ]


def add_summary_sheet(
    wb: Workbook,
    scope: str,
    figures: Sequence[Tuple[str, Path]],
    metadata: pd.DataFrame,
    avg_peaks: pd.DataFrame,
    target_matches: pd.DataFrame,
    nist_status: pd.DataFrame,
    chem_summary: pd.DataFrame,
    pca_var: pd.DataFrame,
) -> None:
    ws = wb.create_sheet("ExecutiveSummary")
    ws["A1"] = f"{scope.upper()} Study Executive Summary"
    ws["A2"] = "Narrative"
    ws["A1"].font = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    ws["A2"].font = Font(name="Calibri", size=12, bold=True, color="1F4E78")

    sample_count, group_count, avg_peak_count, matched, ok_q, fail_q, coverage = parse_metric_counts(
        metadata, avg_peaks, target_matches, nist_status
    )
    top_species = top_species_text(scope, chem_summary)
    pca_sentence = pca_text(pca_var)
    lines = narrative_lines(
        scope,
        sample_count,
        group_count,
        avg_peak_count,
        matched,
        coverage,
        top_species,
        pca_sentence,
    )

    row = 4
    for line in lines:
        ws.cell(row=row, column=1, value=f"- {line}")
        ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
        row += 1

    metric_row = row + 1
    ws.cell(row=metric_row, column=1, value="Key Metrics").font = Font(name="Calibri", size=11, bold=True, color="1F4E78")
    metrics = [
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("NIST Queries OK/Fail", f"{ok_q}/{fail_q}"),
        ("Target Match Coverage", f"{100.0 * coverage:.1f}%"),
    ]
    for i, (k, v) in enumerate(metrics, start=1):
        ws.cell(row=metric_row + i, column=1, value=k).font = Font(name="Calibri", size=10, bold=True)
        ws.cell(row=metric_row + i, column=2, value=v).font = Font(name="Calibri", size=10)

    ws.column_dimensions["A"].width = 86
    ws.column_dimensions["B"].width = 24

    anchors = ["A16", "I16", "A34", "I34"]
    caption_rows = [15, 15, 33, 33]
    for idx, (label, path) in enumerate(figures):
        if idx >= len(anchors):
            break
        ws.cell(row=caption_rows[idx], column=1 if idx % 2 == 0 else 9, value=label).font = Font(
            name="Calibri", size=10, bold=True, color="1F4E78"
        )
        ws.cell(row=caption_rows[idx], column=1 if idx % 2 == 0 else 9).alignment = Alignment(wrap_text=True)

        if XLImage is None:
            ws.cell(row=caption_rows[idx] + 1, column=1 if idx % 2 == 0 else 9, value=f"Image unavailable: {path.name}")
            continue
        try:
            img = XLImage(str(path))
            w = float(getattr(img, "width", 1.0))
            h = float(getattr(img, "height", 1.0))
            max_w = 440.0
            max_h = 220.0
            if w > 0 and h > 0:
                scale = min(max_w / w, max_h / h, 1.0)
                img.width = int(w * scale)
                img.height = int(h * scale)
            ws.add_image(img, anchors[idx])
        except Exception as e:
            ws.cell(row=caption_rows[idx] + 1, column=1 if idx % 2 == 0 else 9, value=f"Could not embed {path.name}: {e}")


def add_trial_parameter_sheets(
    wb: Workbook, scope: str, scope_dir: Path, raw_long: pd.DataFrame, target_matches: pd.DataFrame
) -> None:
    params = ordered_param_sets(scope, raw_long)
    if not params:
        return

    used_names = set(wb.sheetnames)
    for param in params:
        sheet_name = sanitize_sheet_name(str(param), used_names)
        used_names.add(sheet_name)
        ws = wb.create_sheet(sheet_name)
        ws["A1"] = f"Trial Detail: {param}"
        ws["A1"].font = Font(name="Calibri", size=12, bold=True, color="1F4E78")

        d = raw_long[raw_long["param_set"].astype(str) == str(param)].copy() if "param_set" in raw_long.columns else pd.DataFrame()
        row = 3
        ws.cell(row=row, column=1, value="Section A: Trial Summary").font = Font(name="Calibri", size=10, bold=True)
        if d.empty:
            row = write_dataframe(ws, pd.DataFrame([{"note": f"No raw rows for param_set={param}"}]), start_row=row + 1, start_col=1) + 1
        else:
            group_cols = [c for c in ["dataset", "sample_id", "channel"] if c in d.columns]
            if not group_cols:
                group_cols = ["param_set"]
                d["param_set"] = str(param)
            agg = d.groupby(group_cols, dropna=False).agg(
                n_points=("wavelength_nm", "size") if "wavelength_nm" in d.columns else (group_cols[0], "size"),
                wl_min=("wavelength_nm", "min") if "wavelength_nm" in d.columns else (group_cols[0], "size"),
                wl_max=("wavelength_nm", "max") if "wavelength_nm" in d.columns else (group_cols[0], "size"),
                intensity_mean=("irradiance_W_m2_nm", "mean")
                if "irradiance_W_m2_nm" in d.columns
                else (group_cols[0], "size"),
                intensity_max=("irradiance_W_m2_nm", "max")
                if "irradiance_W_m2_nm" in d.columns
                else (group_cols[0], "size"),
            ).reset_index()
            row = write_dataframe(ws, agg, start_row=row + 1, start_col=1) + 1

        ws.cell(row=row, column=1, value="Section B: Target Species Matches").font = Font(name="Calibri", size=10, bold=True)
        tm = pd.DataFrame()
        if not target_matches.empty and "param_set" in target_matches.columns:
            tm = target_matches[target_matches["param_set"].astype(str) == str(param)].copy()
            if "matched" in tm.columns:
                tm = tm[tm["matched"].astype(bool)]
        if tm.empty:
            tm_out = pd.DataFrame([{"note": f"No matched target-species rows for param_set={param}"}])
        else:
            keep = [
                c
                for c in [
                    "dataset",
                    "param_set",
                    "channel",
                    "species",
                    "target_wavelength_nm",
                    "matched_peak_wavelength_nm_0p1",
                    "delta_nm",
                    "matched_peak_intensity",
                ]
                if c in tm.columns
            ]
            tm_out = tm[keep].copy()
            if "delta_nm" in tm_out.columns:
                tm_out["delta_nm"] = pd.to_numeric(tm_out["delta_nm"], errors="coerce")
            tm_out = tm_out.sort_values(
                [c for c in ["dataset", "channel", "species", "target_wavelength_nm"] if c in tm_out.columns],
                ignore_index=True,
            )
        write_dataframe(ws, tm_out.head(200), start_row=row + 1, start_col=1)

        ws.freeze_panes = "A4"
        ws.column_dimensions["A"].width = 18
        ws.column_dimensions["B"].width = 18
        ws.column_dimensions["C"].width = 18
        ws.column_dimensions["D"].width = 14
        ws.column_dimensions["E"].width = 18
        ws.column_dimensions["F"].width = 22
        ws.column_dimensions["G"].width = 12
        ws.column_dimensions["H"].width = 20

        fig_row = 44
        ws.cell(row=fig_row, column=1, value="Section C: Labeled Spectrum Species Chart(s)").font = Font(
            name="Calibri", size=10, bold=True
        )
        pattern = f"*{safe_name(param)}*labeled.png"
        labeled_candidates = sorted((scope_dir / "spectral" / "labels").glob(pattern))
        if not labeled_candidates:
            labeled_candidates = sorted((scope_dir / "spectral" / "labels").glob("*_labeled.png"))
        if labeled_candidates:
            embed_image(
                ws=ws,
                image_path=labeled_candidates[0],
                anchor="A45",
                max_w=520,
                max_h=260,
                missing_cell="A45",
                missing_label=labeled_candidates[0].name,
            )
            if len(labeled_candidates) > 1:
                embed_image(
                    ws=ws,
                    image_path=labeled_candidates[1],
                    anchor="I45",
                    max_w=520,
                    max_h=260,
                    missing_cell="I45",
                    missing_label=labeled_candidates[1].name,
                )
        else:
            ws["A45"] = f"No labeled chart found for param_set={param}"

        ws.cell(row=74, column=1, value="Section D: Concentration Figure Per Trial/Parameter").font = Font(
            name="Calibri", size=10, bold=True
        )
        conc_fig = build_param_concentration_figure(scope_dir=scope_dir, scope=scope, param=str(param), target_matches=target_matches)
        if conc_fig is not None:
            embed_image(
                ws=ws,
                image_path=conc_fig,
                anchor="A75",
                max_w=680,
                max_h=260,
                missing_cell="A75",
                missing_label=conc_fig.name,
            )
        else:
            ws["A75"] = f"No concentration figure could be generated for param_set={param}"


def add_raw_long_sheet(wb: Workbook, raw_long: pd.DataFrame) -> None:
    ws = wb.create_sheet("Raw_Long")
    ws["A1"] = "Raw Long Spectra Data"
    ws["A1"].font = Font(name="Calibri", size=12, bold=True, color="1F4E78")
    end_row = write_dataframe(ws, raw_long, start_row=3, start_col=1)
    ws.freeze_panes = "A4"
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 14
    ws.column_dimensions["E"].width = 16
    ws.column_dimensions["F"].width = 20
    ws.cell(row=end_row + 1, column=1, value=f"Rows exported: {len(raw_long)}").font = Font(name="Calibri", size=10, italic=True)


def add_derived_sheet(
    wb: Workbook,
    pca_var: pd.DataFrame,
    pca_scores: pd.DataFrame,
    target_summary: pd.DataFrame,
    peak_matches: pd.DataFrame,
) -> None:
    ws = wb.create_sheet("Derived_Key")
    ws["A1"] = "Key Derived Data: PCA + Target Species Band Matching"
    ws["A1"].font = Font(name="Calibri", size=12, bold=True, color="1F4E78")

    row = 3
    ws.cell(row=row, column=1, value="Section A: PCA Explained Variance").font = Font(name="Calibri", size=10, bold=True)
    row = write_dataframe(ws, pca_var, start_row=row + 1, start_col=1) + 1

    ws.cell(row=row, column=1, value="Section B: PCA Scores").font = Font(name="Calibri", size=10, bold=True)
    pca_scores_small = pca_scores.head(120)
    row = write_dataframe(ws, pca_scores_small, start_row=row + 1, start_col=1) + 1

    ws.cell(row=row, column=1, value="Section C: Target Match Summary").font = Font(name="Calibri", size=10, bold=True)
    row = write_dataframe(ws, target_summary, start_row=row + 1, start_col=1) + 1

    ws.cell(row=row, column=1, value="Section D: Top Peak-to-Species Matches").font = Font(name="Calibri", size=10, bold=True)
    write_dataframe(ws, peak_matches, start_row=row + 1, start_col=1)
    ws.freeze_panes = "A4"
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 20
    ws.column_dimensions["C"].width = 20
    ws.column_dimensions["D"].width = 18
    ws.column_dimensions["E"].width = 18
    ws.column_dimensions["F"].width = 18
    ws.column_dimensions["G"].width = 18


def build_peak_match_table(target_matches: pd.DataFrame) -> pd.DataFrame:
    if target_matches.empty:
        return pd.DataFrame([{"note": "No target species matches available."}])
    d = target_matches.copy()
    if "matched" in d.columns:
        d = d[d["matched"].astype(bool)].copy()
    keep = [
        c
        for c in [
            "dataset",
            "param_set",
            "channel",
            "species",
            "target_wavelength_nm",
            "matched_peak_rank",
            "matched_peak_wavelength_nm_0p1",
            "matched_peak_intensity",
            "delta_nm",
        ]
        if c in d.columns
    ]
    if not keep:
        return pd.DataFrame([{"note": "Target match schema missing expected columns."}])
    d = d[keep].copy()
    if "delta_nm" in d.columns:
        d["delta_nm"] = pd.to_numeric(d["delta_nm"], errors="coerce")
        d = d.sort_values("delta_nm", ascending=True, ignore_index=True)
    return d.head(120)


def add_notes_sheet(wb: Workbook, scope_dir: Path, nist_status: pd.DataFrame) -> None:
    ws = wb.create_sheet("Notes")
    ws["A1"] = "Scope Notes and Traceability"
    ws["A1"].font = Font(name="Calibri", size=12, bold=True, color="1F4E78")
    ws["A3"] = "NIST data source"
    ws["A4"] = "https://www.nist.gov/pml/atomic-spectra-database"
    ws["A5"] = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
    ws["A7"] = "Target species lines:"
    ws["A8"] = "configs/target_species_lines.csv"
    ws["A10"] = "Workbook uses final scoped outputs under:"
    ws["A11"] = scope_dir.as_posix()

    ws["A13"] = "NIST fetch status"
    ws["A13"].font = Font(name="Calibri", size=10, bold=True)
    write_dataframe(ws, nist_status, start_row=14, start_col=1)
    ws.column_dimensions["A"].width = 52
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 72


def build_scope_workbook(scope: str) -> Path:
    scope_dir = OUTPUT_ROOT / scope
    if not scope_dir.exists():
        raise FileNotFoundError(f"Missing scope directory: {scope_dir}")

    raw_long = read_csv_or_note(scope_dir / "spectral/base/raw/spectra_long.csv")
    metadata = read_csv_or_note(scope_dir / "spectral/base/raw/metadata.csv")
    avg_peaks = read_csv_or_note(scope_dir / "spectral/base/raw/averaged_peaks_top10.csv")
    target_matches = read_csv_or_note(scope_dir / "spectral/base/raw/target_species_peak_matches.csv")
    nist_status = read_csv_or_note(scope_dir / "spectral/base/raw/nist_fetch_status.csv")
    target_summary = read_csv_or_note(scope_dir / "spectral/base/raw/target_species_match_summary.csv")
    chem_summary = read_csv_or_note(scope_dir / "chemspecies/csv/dataset_species_concentration_summary.csv")
    pca_var = read_csv_or_note(scope_dir / "pca/pca_explained_variance.csv")
    pca_scores = read_csv_or_note(scope_dir / "pca/pca_scores.csv")
    key_figures = select_key_figures(scope_dir, scope)
    peak_matches = build_peak_match_table(target_matches)

    wb = Workbook()
    wb.remove(wb.active)
    add_summary_sheet(
        wb=wb,
        scope=scope,
        figures=key_figures,
        metadata=metadata,
        avg_peaks=avg_peaks,
        target_matches=target_matches,
        nist_status=nist_status,
        chem_summary=chem_summary,
        pca_var=pca_var,
    )
    add_trial_parameter_sheets(
        wb=wb,
        scope=scope,
        scope_dir=scope_dir,
        raw_long=raw_long,
        target_matches=target_matches,
    )
    add_raw_long_sheet(wb=wb, raw_long=raw_long)
    add_derived_sheet(
        wb=wb,
        pca_var=pca_var,
        pca_scores=pca_scores,
        target_summary=target_summary,
        peak_matches=peak_matches,
    )
    add_notes_sheet(wb=wb, scope_dir=scope_dir, nist_status=nist_status)

    workbook_path = scope_dir / f"{scope}_executive_report.xlsx"
    fallback = scope_dir / f"{scope}_executive_report_new.xlsx"
    try:
        wb.save(workbook_path)
        if fallback.exists():
            try:
                fallback.unlink()
            except OSError:
                pass
        return workbook_path
    except PermissionError:
        if workbook_path.exists():
            return workbook_path
        wb.save(fallback)
        return fallback


def main() -> int:
    generated: List[Path] = []
    failures: List[str] = []
    for scope in SCOPES:
        try:
            generated.append(build_scope_workbook(scope))
        except Exception as e:
            failures.append(f"{scope}: {e}")

    if generated:
        print("Wrote executive workbooks:")
        for path in generated:
            print(f"  {path}")
    if failures:
        print("Executive workbook generation failures:")
        for row in failures:
            print(f"  {row}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
