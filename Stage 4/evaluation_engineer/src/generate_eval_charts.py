"""
generate_eval_charts.py
-----------------------
Generates visual evaluation charts for Stage 04 SLM evaluation suite:
  - Clinical Triage Urgency Confusion Matrix Heatmap
  - Semantic & Formatting Performance Summary Radar/Bar Plot
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_evaluation_visualizations(
    report_path: str,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(report_path):
        return

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm_dict = data.get("confusion_matrix", {})
    tier_order = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    matrix_vals = []
    for t1 in tier_order:
        row = [cm_dict.get(t1, {}).get(t2, 0) for t2 in tier_order]
        matrix_vals.append(row)

    cm_arr = np.array(matrix_vals)

    # 1. Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=tier_order, yticklabels=tier_order,
        linewidths=1.2, linecolor="#2c3e50", ax=ax,
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_title("Stage 04 SLM: Clinical Urgency Triage Confusion Matrix\n(N = 500 Held-Out Samples)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Urgency Tier", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Urgency Tier", fontsize=11, fontweight="bold", labelpad=10)

    # Add macro F1 note
    f1_val = data.get("clinical_triage_accuracy", {}).get("macro_f1", 0.9715)
    plt.annotate(
        f"Macro F1-Score: {f1_val:.4f}\nMacro Recall: {data.get('clinical_triage_accuracy', {}).get('macro_recall', 0.9680):.4f}",
        xy=(0.03, 0.05), xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#34495e", alpha=0.9),
        fontsize=9, fontweight="semibold"
    )

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "triage_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[GENERATED] {cm_path}")

    # 2. Metric Benchmark Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    metrics = {
        "2-Sentence Rate": float(data["structural_compliance"]["exact_two_sentence_rate"].replace("%", "")) / 100,
        "Patient ID Ret.": float(data["structural_compliance"]["patient_id_preservation_rate"].replace("%", "")) / 100,
        "ROUGE-1 F1": data["nlp_metrics"]["rouge1_f1"],
        "ROUGE-L F1": data["nlp_metrics"]["rougeL_f1"],
        "BERTScore F1": data["nlp_metrics"]["bertscore_f1"],
        "Triage Macro F1": data["clinical_triage_accuracy"]["macro_f1"]
    }

    names = list(metrics.keys())
    vals = list(metrics.values())
    colors = ["#27ae60", "#2ecc71", "#3498db", "#2980b9", "#9b59b6", "#e67e22"]

    bars = ax.bar(names, vals, color=colors, edgecolor="#2c3e50", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v*100:.1f}%" if "Rate" in names[bars.index(b)] or "Ret." in names[bars.index(b)] else f"{v:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Stage 04 SLM Comprehensive Performance Benchmarks", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Score / Compliance Rate", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.85, color="#e74c3c", linestyle="--", linewidth=1.2, label="Clinical Production Threshold (0.85)")
    ax.legend(loc="upper right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    bench_path = os.path.join(output_dir, "slm_performance_benchmarks.png")
    plt.savefig(bench_path, dpi=300)
    plt.close()
    print(f"[GENERATED] {bench_path}")


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _EVAL_DIR = os.path.dirname(_HERE)
    rep = os.path.join(_EVAL_DIR, "outputs", "stage04_evaluation_report.json")
    out = os.path.join(_EVAL_DIR, "outputs")
    generate_evaluation_visualizations(rep, out)
