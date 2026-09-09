"""
================================================================================
STAGE 02 MULTI-MODAL CANCER PROGRESSION PIPELINE - EDA MODULE
File: 02_eda_analysis.py
Role: EDA Engineer
================================================================================
Automated Exploratory Data Analysis & Validation Script for multi-modal cancer
progression trajectory data. Performs data integrity checks, numerical statistics,
image existence verification, progression rate calculation, and outputs a 
comprehensive summary report + visualisation charts.
"""

import os
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE_CANCER = [
    "#4361EE", "#3A0CA3", "#7209B7", "#F72585",
    "#4CC9F0", "#4895EF", "#560BAD", "#B5179E"
]
PALETTE_LABEL  = ["#4CC9F0", "#F72585"]   # 0 = blue, 1 = pink
DARK_BG        = "#0F0F1A"
CARD_BG        = "#1A1A2E"


def _save(fig: plt.Figure, path: str):
    """Save figure with tight layout and dark facecolor."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   Chart saved -> {path}")


def generate_eda_charts(df: pd.DataFrame, charts_dir: str):
    """
    Generates and saves 6 EDA visualisation charts into charts_dir.
    """
    os.makedirs(charts_dir, exist_ok=True)
    num_cols = ["ctDNA_vaf_pct", "ct_tumor_vol_cm3", "biomarker_ng_ml", "wsi_atypia_score"]

    print("\n--- Generating EDA Charts ---")

    # ── Chart 1: Class Balance Pie ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    counts = df["label_progression"].value_counts().sort_index()
    labels = [f"No Progression\n({counts[0]:,})", f"Progression\n({counts[1]:,})"]
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=PALETTE_LABEL,
        startangle=90,
        wedgeprops=dict(edgecolor=DARK_BG, linewidth=2),
        textprops=dict(color="white", fontsize=12)
    )
    for at in autotexts:
        at.set_fontsize(14)
        at.set_fontweight("bold")
    ax.set_title("Class Balance: Progression Labels", color="white", fontsize=14, pad=20)
    _save(fig, os.path.join(charts_dir, "01_class_balance.png"))

    # ── Chart 2: Progression Rate by Cancer Type ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    prog_rate = (
        df.groupby("cancer_type")["label_progression"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    bars = ax.bar(
        prog_rate["cancer_type"],
        prog_rate["label_progression"],
        color=PALETTE_CANCER[:len(prog_rate)],
        edgecolor="none",
        width=0.6
    )
    # Value labels on bars
    for bar, val in zip(bars, prog_rate["label_progression"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center", va="bottom", color="white", fontsize=10, fontweight="bold"
        )
    # Overall average line
    overall = df["label_progression"].mean() * 100
    ax.axhline(overall, color="#F72585", linestyle="--", linewidth=1.5, label=f"Overall avg: {overall:.1f}%")
    ax.legend(facecolor=CARD_BG, edgecolor="none", labelcolor="white", fontsize=10)
    ax.set_xlabel("Cancer Type", color="white", fontsize=12)
    ax.set_ylabel("Progression Rate (%)", color="white", fontsize=12)
    ax.set_title("Progression Rate by Cancer Type", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.set_ylim(0, prog_rate["label_progression"].max() + 8)
    _save(fig, os.path.join(charts_dir, "02_progression_rate_by_cancer.png"))

    # ── Chart 3: Feature Distributions (KDE) ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=DARK_BG)
    fig.suptitle("Feature Distributions (by Progression Label)", color="white", fontsize=14, y=1.01)
    labels_map = {0: "No Progression", 1: "Progression"}
    for ax, col in zip(axes.flat, num_cols):
        ax.set_facecolor(CARD_BG)
        for lbl, color in zip([0, 1], PALETTE_LABEL):
            subset = df[df["label_progression"] == lbl][col]
            subset.plot.kde(ax=ax, color=color, linewidth=2, label=labels_map[lbl])
            ax.fill_between(
                np.linspace(subset.min(), subset.max(), 200),
                0, 0,    # placeholder — seaborn kdeplot handles fill better
                alpha=0.0
            )
        ax.set_title(col.replace("_", " ").title(), color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
        ax.set_xlabel("", color="white")
        ax.set_ylabel("Density", color="white", fontsize=9)
        legend = ax.legend(facecolor=CARD_BG, edgecolor="none", labelcolor="white", fontsize=9)
    plt.tight_layout()
    _save(fig, os.path.join(charts_dir, "03_feature_distributions.png"))

    # ── Chart 4: Boxplots by Label ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("Feature Values: No Progression vs Progression", color="white", fontsize=13)
    for ax, col in zip(axes, num_cols):
        ax.set_facecolor(CARD_BG)
        data_0 = df[df["label_progression"] == 0][col].values
        data_1 = df[df["label_progression"] == 1][col].values
        bp = ax.boxplot(
            [data_0, data_1],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="white"),
            capprops=dict(color="white"),
            flierprops=dict(markerfacecolor="white", marker="o", markersize=3, alpha=0.4)
        )
        for patch, color in zip(bp["boxes"], PALETTE_LABEL):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No Prog.", "Prog."], color="white", fontsize=9)
        ax.set_title(col.replace("_", " ").title(), color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
    plt.tight_layout()
    _save(fig, os.path.join(charts_dir, "04_feature_boxplots_by_label.png"))

    # ── Chart 5: Correlation Heatmap ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    corr_cols = num_cols + ["label_progression"]
    corr = df[corr_cols].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
        linewidths=0.5, linecolor=DARK_BG,
        ax=ax,
        annot_kws={"size": 9, "color": "white"},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")
    _save(fig, os.path.join(charts_dir, "05_correlation_heatmap.png"))

    # ── Chart 6: Mean Feature Trajectory Over Timepoints ─────────────────────
    if "sequence_step" in df.columns and "timepoint_month" in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=DARK_BG)
        fig.suptitle("Mean Feature Trajectory Over Timepoints (0→1 = Progression)", color="white", fontsize=13)
        for ax, col in zip(axes.flat, num_cols):
            ax.set_facecolor(CARD_BG)
            for lbl, color, lname in zip([0, 1], PALETTE_LABEL, ["No Progression", "Progression"]):
                grp = (
                    df[df["label_progression"] == lbl]
                    .groupby("sequence_step")[col]
                    .mean()
                )
                ax.plot(grp.index, grp.values, color=color, linewidth=2.5,
                        marker="o", markersize=6, label=lname)
                ax.fill_between(grp.index, grp.values, alpha=0.12, color=color)
            ax.set_title(col.replace("_", " ").title(), color="white", fontsize=10)
            ax.set_xlabel("Sequence Step", color="white", fontsize=9)
            ax.tick_params(colors="white")
            ax.spines[:].set_visible(False)
            ax.legend(facecolor=CARD_BG, edgecolor="none", labelcolor="white", fontsize=8)
        plt.tight_layout()
        _save(fig, os.path.join(charts_dir, "06_feature_trajectory_over_time.png"))

    print(f"   All charts saved to: {charts_dir}\n")


def run_eda_pipeline(
    csv_path: str,
    img_dir: str,
    output_report_path: str = "eda_summary_report.txt"
) -> Dict[str, Any]:
    """
    Executes automated EDA analysis, generates summary report + charts.
    """
    print(f"============================================================")
    print(f"STARTING STAGE 02 AUTOMATED EDA & VALIDATION PIPELINE")
    print(f"============================================================")
    print(f"CSV Path   : {csv_path}")
    print(f"Images Dir : {img_dir}")
    print(f"Output File: {output_report_path}\n")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    report_lines = []

    def log(msg: str):
        print(msg)
        report_lines.append(msg)

    log("============================================================")
    log("STAGE 02 MULTI-MODAL CANCER PROGRESSION - EDA SUMMARY REPORT")
    log("============================================================\n")

    # 1. OVERVIEW & SHAPE
    n_rows = len(df)
    n_patients = df["patient_id"].nunique()
    log("1. DATASET OVERVIEW")
    log(f"   - Total Sequence Rows  : {n_rows}")
    log(f"   - Unique Patients      : {n_patients}")
    log(f"   - Cancer Types Count   : {df['cancer_type'].nunique()}")
    log(f"   - Cancer Types List    : {sorted(df['cancer_type'].unique().tolist())}\n")

    # 2. DATA INTEGRITY & MISSING VALUE CHECKS
    log("2. DATA INTEGRITY CHECKS")
    missing_vals = df.isnull().sum()
    log(f"   - Missing Values Count :")
    for col, count in missing_vals.items():
        log(f"       * {col}: {count}")

    # Sequence Continuity Check (5 timepoints per patient)
    p_counts = df.groupby("patient_id")["sequence_step"].count()
    invalid_p_seq = p_counts[p_counts != 5]
    log(f"   - Sequence Continuity Verification (5 timepoints/patient):")
    if len(invalid_p_seq) == 0:
        log("       [PASS] All patients have exactly 5 consecutive timepoints (steps 0 to 4).")
    else:
        log(f"       [FAIL] {len(invalid_p_seq)} patients have invalid step counts!")

    # Step range check
    step_range_valid = (df.groupby("patient_id")["sequence_step"].apply(list).apply(lambda s: sorted(s) == [0, 1, 2, 3, 4])).all()
    log(f"   - Step Index Integrity (0, 1, 2, 3, 4): {'[PASS]' if step_range_valid else '[FAIL]'}\n")

    # 3. NUMERICAL FEATURE SUMMARY STATISTICS
    num_cols = ["ctDNA_vaf_pct", "ct_tumor_vol_cm3", "biomarker_ng_ml", "wsi_atypia_score"]
    log("3. NUMERICAL FEATURE SUMMARY STATISTICS")
    stats_df = df[num_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    stats_df.rename(columns={"50%": "median"}, inplace=True)
    log(stats_df.to_string())
    log("\n   --- Numerical Features Grouped by Progression Label (0 vs 1) ---")
    grouped_label = df.groupby("label_progression")[num_cols].mean()
    log(grouped_label.to_string())
    log("\n")

    # 4. IMAGE FILE EXISTENCE VERIFICATION
    log("4. IMAGE FILE EXISTENCE VERIFICATION")
    missing_pathology = 0
    missing_ct = 0

    for _, row in df.iterrows():
        h_path = os.path.join(img_dir, row["pathology_img_file"])
        if not os.path.exists(h_path):
            basename = os.path.basename(row["pathology_img_file"])
            alt = os.path.join(img_dir, row["cancer_type"].lower(), basename)
            if not os.path.exists(alt):
                missing_pathology += 1

        c_path = os.path.join(img_dir, row["ct_img_file"])
        if not os.path.exists(c_path):
            basename = os.path.basename(row["ct_img_file"])
            alt = os.path.join(img_dir, row["cancer_type"].lower(), basename)
            if not os.path.exists(alt):
                missing_ct += 1

    log(f"   - Missing Pathology Images : {missing_pathology} / {n_rows}")
    log(f"   - Missing 2D CT Images     : {missing_ct} / {n_rows}")
    if missing_pathology == 0 and missing_ct == 0:
        log("   [PASS] 100% of histology and CT image files verified in disk!")
    else:
        log(f"   [WARNING] Missing image files detected.")
    log("\n")

    # 5. CANCER TYPE SUBGROUP PROGRESSION DISTRIBUTION
    log("5. CANCER TYPE SUBGROUP PROGRESSION DISTRIBUTION")
    log(f"{'Cancer Type':<12} | {'Total Samples':<14} | {'Progression (1)':<16} | {'Progression Rate (%)':<20}")
    log("-" * 70)
    subgroup_summary = {}

    for c_type, group in df.groupby("cancer_type"):
        total = len(group)
        pos = int(group["label_progression"].sum())
        rate = (pos / total) * 100.0
        subgroup_summary[c_type] = {"total": total, "progression_count": pos, "rate_pct": round(rate, 2)}
        log(f"{c_type:<12} | {total:<14} | {pos:<16} | {rate:<20.2f}")

    total_prog_rate = (df["label_progression"].sum() / n_rows) * 100.0
    log("-" * 70)
    log(f"{'OVERALL':<12} | {n_rows:<14} | {int(df['label_progression'].sum()):<16} | {total_prog_rate:<20.2f}\n")

    # WRITE REPORT TO FILE
    output_dir = os.path.dirname(output_report_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"EDA Summary Report saved successfully to '{output_report_path}'.")

    # 6. GENERATE CHARTS
    charts_dir = os.path.join(os.path.dirname(output_report_path), "eda_charts")
    generate_eda_charts(df, charts_dir)

    return {
        "n_rows": n_rows,
        "n_patients": n_patients,
        "missing_vals": missing_vals.to_dict(),
        "subgroup_summary": subgroup_summary,
        "charts_dir": charts_dir
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "data", "stage02_lstm_8types_3000samples.csv")
    if not os.path.exists(csv_file):
        csv_file = os.path.join(script_dir, "stage02_lstm_8types_3000samples.csv")
        
    images_dir = os.path.join(script_dir, "2d_images_large")
    report_file = os.path.join(script_dir, "eda_summary_report.txt")

    run_eda_pipeline(csv_file, images_dir, report_file)
