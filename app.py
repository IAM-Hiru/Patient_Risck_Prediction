"""
================================================================================
ONCOLOGY MEDICINE PREDICTION — STREAMLIT DASHBOARD
app.py  |  Entry Point
================================================================================
Professional multi-page dashboard covering:
  • Home / Overview
  • Stage 1 — ML Model Comparison (classical models)
  • Stage 2 — EDA Analysis + Charts
  • Stage 2 — Deep Learning Evaluation Metrics
  • Stage 2 — Live Patient Inference
================================================================================
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Resolve project root ────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
STAGE1_DIR  = ROOT / "Stage1"
STAGE2_DIR  = ROOT / "Stage2"
CHARTS_DIR  = STAGE2_DIR / "eda_charts"
CSV_PATH    = STAGE2_DIR / "data" / "stage02_lstm_8types_3000samples.csv"
IMG_DIR     = STAGE2_DIR / "2d_images_large"
CKPT_PATH   = STAGE2_DIR / "best_stage02_multimodal_lstm.pth"
EVAL_JSON   = STAGE2_DIR / "evaluation_results.json"
COMP_CSV    = STAGE1_DIR / "model_comparison_tuned.csv"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OncoPred · AI Cancer Progression",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Import Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #0A0A14;
    --bg-card:      #12121F;
    --bg-card2:     #1A1A2E;
    --accent-blue:  #4361EE;
    --accent-pink:  #F72585;
    --accent-cyan:  #4CC9F0;
    --accent-purple:#7209B7;
    --text-primary: #F0F0FF;
    --text-muted:   #8A8AB0;
    --border:       rgba(255,255,255,0.07);
    --radius:       14px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Metric cards */
.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(67,97,238,0.2);
}
.metric-card .label {
    font-size: 12px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .sub {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Section header */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 8px 0 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--accent-blue);
}

/* Risk badge */
.risk-low    { background:#0d3b2e; color:#4ade80; border:1px solid #4ade80; padding:4px 14px; border-radius:20px; font-weight:600; }
.risk-mod    { background:#3b2a0d; color:#fbbf24; border:1px solid #fbbf24; padding:4px 14px; border-radius:20px; font-weight:600; }
.risk-high   { background:#3b0d0d; color:#f87171; border:1px solid #f87171; padding:4px 14px; border-radius:20px; font-weight:600; }

/* Table */
.styled-table {
    width: 100%; border-collapse: collapse;
    font-size: 13px; color: var(--text-primary);
}
.styled-table th {
    background: var(--bg-card2);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    color: var(--accent-cyan);
    border-bottom: 2px solid var(--border);
}
.styled-table td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
}
.styled-table tr:hover td { background: rgba(67,97,238,0.06); }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0A0A14 0%, #12121F 50%, #1A0A2E 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 70% 40%, rgba(67,97,238,0.08), transparent 55%),
                radial-gradient(circle at 20% 80%, rgba(247,37,133,0.06), transparent 50%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px; font-weight: 700;
    background: linear-gradient(135deg, #fff 30%, var(--accent-cyan));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 12px;
}
.hero p { color: var(--text-muted); font-size: 16px; max-width: 600px; margin: 0; line-height: 1.7; }

.pill {
    display: inline-block;
    background: rgba(67,97,238,0.15);
    color: var(--accent-cyan);
    border: 1px solid rgba(76,201,240,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    margin: 4px 4px 12px 0;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:42px;'>🧬</div>
        <div style='font-family:"Space Grotesk",sans-serif; font-size:18px; font-weight:700;
                    background:linear-gradient(135deg,#4CC9F0,#4361EE);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            OncoPred AI
        </div>
        <div style='font-size:11px; color:#8A8AB0; margin-top:4px;'>Multi-Modal Cancer Progression</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Overview",
         "📊  Stage 1 · ML Comparison",
         "🔬  Stage 2 · EDA Analysis",
         "📈  Stage 2 · Model Evaluation",
         "🤖  Stage 2 · Live Inference"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#8A8AB0; padding:8px 0; line-height:1.8;'>
        <b style='color:#F0F0FF;'>Pipeline Stages</b><br>
        Stage 1 · Classical ML (toxicity)<br>
        Stage 2 · Deep Learning (progression)<br><br>
        <b style='color:#F0F0FF;'>Architecture</b><br>
        ResNet-18 + Bi-LSTM<br>
        Multi-modal (Path + CT + Tabular)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def risk_badge(tier: str) -> str:
    cls = {"Low Progression Risk": "risk-low",
           "Moderate Risk": "risk-mod",
           "High Progression Risk": "risk-high"}.get(tier, "risk-low")
    return f'<span class="{cls}">{tier}</span>'


def dark_fig():
    fig, ax = plt.subplots(facecolor="#12121F")
    ax.set_facecolor("#1A1A2E")
    return fig, ax


def dark_figs(rows, cols, **kwargs):
    fig, axes = plt.subplots(rows, cols, facecolor="#12121F", **kwargs)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor("#1A1A2E")
    return fig, axes


PALETTE = ["#4361EE","#3A0CA3","#7209B7","#F72585","#4CC9F0","#4895EF","#560BAD","#B5179E"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class="hero">
        <h1>OncoPred AI Dashboard</h1>
        <p>A multi-stage AI pipeline for cancer progression prediction combining classical
        machine learning and multi-modal deep learning on longitudinal patient trajectories.</p>
        <div style='margin-top:20px;'>
            <span class="pill">ResNet-18 Pathology Encoder</span>
            <span class="pill">CNN CT Encoder</span>
            <span class="pill">Bidirectional LSTM</span>
            <span class="pill">8 Cancer Types</span>
            <span class="pill">600 Patients · 5 Timepoints</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "Patients", "600", "3,000 sequence rows"),
        (col2, "Cancer Types", "8",  "BLCA · BRCA · COAD · LUAD · LUSC · PRAD · SKCM · STAD"),
        (col3, "Timepoints", "5",   "M0 → M3 → M6 → M9 → M12"),
        (col4, "Model",  "Bi-LSTM", "ResNet-18 + CT-CNN + LSTM"),
    ]
    for col, label, value, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Stage 1 — Classical ML</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#12121F; border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:20px;
                    line-height:2; font-size:14px; color:#C0C0E0;'>
        🔹 <b>Task:</b> Drug toxicity prediction (binary classification)<br>
        🔹 <b>Models:</b> Decision Tree, Random Forest, XGBoost, Extra Trees, Stacking Ensemble<br>
        🔹 <b>Dataset:</b> Oncology risk dataset (2,000 samples)<br>
        🔹 <b>Output:</b> Best tuned model saved as .pkl checkpoint
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Stage 2 — Deep Learning</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#12121F; border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:20px;
                    line-height:2; font-size:14px; color:#C0C0E0;'>
        🔹 <b>Task:</b> Longitudinal cancer progression (binary, per-timepoint)<br>
        🔹 <b>Inputs:</b> Histopathology tile + 2D CT slice + 4 biomarkers per timepoint<br>
        🔹 <b>Architecture:</b> ResNet-18 PathologyEncoder + CTEncoder + Bi-LSTM<br>
        🔹 <b>Output:</b> Per-step progression probabilities + clinical risk tier
        </div>
        """, unsafe_allow_html=True)

    # Pipeline flow diagram (simple)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 3), facecolor="#0A0A14")
    ax.set_facecolor("#0A0A14")
    ax.axis("off")

    boxes = [
        (0.05, "Histopathology\nImage (224×224)", "#4361EE"),
        (0.20, "ResNet-18\nPathology Encoder\n→ 128-dim", "#3A0CA3"),
        (0.40, "Bidirectional\nLSTM\n(2 layers, 256-dim)", "#7209B7"),
        (0.60, "Classification\nHead\n(Linear + Sigmoid)", "#F72585"),
        (0.78, "Per-Timepoint\nProgression\nProbability", "#4CC9F0"),
    ]
    boxes2 = [
        (0.05, "2D CT Scan\n(224×224 grey)", "#4895EF"),
        (0.20, "CNN CT Encoder\n→ 128-dim", "#560BAD"),
    ]
    boxes3 = [
        (0.05, "4 Biomarkers\n(ctDNA, Tumor Vol\nbiomarker, atypia)", "#B5179E"),
        (0.20, "StandardScaler\n→ 4-dim", "#4361EE"),
    ]

    y_positions = [0.72, 0.45, 0.18]
    all_boxes = [boxes, boxes2, boxes3]
    for row_idx, (row_boxes, y_center) in enumerate(zip(all_boxes, y_positions)):
        for x, label, color in row_boxes:
            rect = mpatches.FancyBboxPatch(
                (x, y_center - 0.18), 0.13, 0.32,
                boxstyle="round,pad=0.01", linewidth=1.5,
                edgecolor=color, facecolor=color + "22"
            )
            ax.add_patch(rect)
            ax.text(x + 0.065, y_center, label, ha="center", va="center",
                    color="white", fontsize=7, fontweight="500",
                    multialignment="center")

    # Arrows
    arrow_kwargs = dict(arrowstyle="-|>", color="#4CC9F0", lw=1.5,
                        connectionstyle="arc3,rad=0")
    for y in y_positions:
        for x_start in [0.19, 0.34, 0.54, 0.74]:
            ax.annotate("", xy=(x_start + 0.01, y), xytext=(x_start - 0.01, y),
                        arrowprops=dict(arrowstyle="-|>", color="#4CC9F0", lw=1.3))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STAGE 1 ML COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif "Stage 1" in page:
    st.markdown('<div class="section-header">Stage 1 · Classical ML Model Comparison</div>',
                unsafe_allow_html=True)

    if not COMP_CSV.exists():
        st.warning(f"Model comparison file not found at: `{COMP_CSV}`")
        st.stop()

    df_comp = pd.read_csv(COMP_CSV)
    df_comp.columns = [c.strip() for c in df_comp.columns]

    # Detect metric columns
    metric_cols = [c for c in df_comp.columns if c.lower() not in
                   ["model", "model_name", "name", "classifier"]]
    model_col   = [c for c in df_comp.columns if c.lower() in
                   ["model", "model_name", "name", "classifier"]][0]

    # Best model highlight
    if "ROC_AUC" in df_comp.columns or "roc_auc" in df_comp.columns:
        auc_col = "ROC_AUC" if "ROC_AUC" in df_comp.columns else "roc_auc"
        best_row = df_comp.loc[df_comp[auc_col].idxmax()]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Best Model</div>
                <div class="value" style="font-size:22px;">{best_row[model_col]}</div>
                <div class="sub">Highest ROC-AUC</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="label">ROC-AUC</div>
                <div class="value">{best_row[auc_col]:.4f}</div>
                <div class="sub">Test performance</div></div>""", unsafe_allow_html=True)
        with c3:
            f1_col = next((c for c in df_comp.columns if "f1" in c.lower()), None)
            val = f"{best_row[f1_col]:.4f}" if f1_col else "N/A"
            st.markdown(f"""<div class="metric-card">
                <div class="label">F1 Score</div>
                <div class="value">{val}</div>
                <div class="sub">Best model F1</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table
    st.markdown('<div class="section-header">All Models</div>', unsafe_allow_html=True)
    html_rows = "".join(
        f"<tr>{''.join(f'<td>{v:.4f}</td>' if isinstance(v, float) else f'<td>{v}</td>' for v in row)}</tr>"
        for _, row in df_comp.iterrows()
    )
    headers = "".join(f"<th>{c}</th>" for c in df_comp.columns)
    st.markdown(f'<table class="styled-table"><thead><tr>{headers}</tr></thead><tbody>{html_rows}</tbody></table>',
                unsafe_allow_html=True)

    # Bar chart per metric
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Performance Comparison</div>', unsafe_allow_html=True)
    num_metrics = len(metric_cols)
    if num_metrics:
        cols = st.columns(min(num_metrics, 3))
        for i, m_col in enumerate(metric_cols):
            with cols[i % len(cols)]:
                fig, ax = dark_fig()
                colors = PALETTE[:len(df_comp)]
                ax.barh(df_comp[model_col], df_comp[m_col], color=colors, edgecolor="none")
                ax.set_title(m_col, color="white", fontsize=11, fontweight="600")
                ax.tick_params(colors="white", labelsize=8)
                ax.spines[:].set_visible(False)
                ax.set_xlim(df_comp[m_col].min() * 0.95, df_comp[m_col].max() * 1.03)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown('<div class="section-header">Stage 2 · Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    if not CSV_PATH.exists():
        st.error(f"Dataset CSV not found: `{CSV_PATH}`")
        st.stop()

    df = pd.read_csv(CSV_PATH)

    # Quick stats row
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "Total Rows",       f"{len(df):,}",                       "Sequence rows"),
        (c2, "Unique Patients",  f"{df['patient_id'].nunique():,}",     "600 patients"),
        (c3, "Progression Rate", f"{df['label_progression'].mean()*100:.1f}%", "Overall"),
        (c4, "Missing Values",   f"{df.isnull().sum().sum()}",          "Across all columns"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Show saved charts if available ──────────────────────────────────────
    chart_files = sorted(CHARTS_DIR.glob("*.png")) if CHARTS_DIR.exists() else []
    if chart_files:
        st.markdown('<div class="section-header">EDA Charts</div>', unsafe_allow_html=True)
        chart_titles = {
            "01_class_balance.png":              "Class Balance",
            "02_progression_rate_by_cancer.png": "Progression Rate by Cancer Type",
            "03_feature_distributions.png":      "Feature Distributions (KDE)",
            "04_feature_boxplots_by_label.png":  "Feature Boxplots by Label",
            "05_correlation_heatmap.png":         "Correlation Heatmap",
            "06_feature_trajectory_over_time.png":"Feature Trajectory Over Time",
        }
        # Pair charts into 2 columns
        pairs = [(chart_files[i], chart_files[i+1] if i+1 < len(chart_files) else None)
                 for i in range(0, len(chart_files), 2)]
        for left, right in pairs:
            col_l, col_r = st.columns(2)
            with col_l:
                title = chart_titles.get(left.name, left.stem.replace("_", " ").title())
                st.markdown(f"**{title}**")
                st.image(str(left), use_container_width=True)
            if right:
                with col_r:
                    title = chart_titles.get(right.name, right.stem.replace("_", " ").title())
                    st.markdown(f"**{title}**")
                    st.image(str(right), use_container_width=True)
    else:
        st.info("EDA charts not found. Run `02_eda_analysis.py` to generate them.")

    # ── Numerical stats table ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Summary Statistics</div>', unsafe_allow_html=True)
    num_cols = ["ctDNA_vaf_pct", "ct_tumor_vol_cm3", "biomarker_ng_ml", "wsi_atypia_score"]
    stats = df[num_cols].describe().T[["mean","std","min","50%","max"]].rename(columns={"50%":"median"})
    st.dataframe(stats.style.background_gradient(cmap="Blues", axis=1),
                 use_container_width=True)

    # ── Subgroup table ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Cancer Type Subgroup Breakdown</div>', unsafe_allow_html=True)
    grp = df.groupby("cancer_type").agg(
        Total=("label_progression","count"),
        Progressors=("label_progression","sum")
    ).reset_index()
    grp["Progression Rate (%)"] = (grp["Progressors"] / grp["Total"] * 100).round(2)
    st.dataframe(grp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Evaluation" in page:
    st.markdown('<div class="section-header">Stage 2 · Deep Learning Model Evaluation</div>',
                unsafe_allow_html=True)

    if not EVAL_JSON.exists():
        st.warning("evaluation_results.json not found. Run `04_model_evaluation.py` first.")
        st.stop()

    with open(EVAL_JSON) as f:
        results = json.load(f)

    gm = results["global_metrics"]
    sg = results["subgroup_metrics"]

    # Global metric cards
    keys = ["ROC_AUC", "PR_AUC", "F1_Score", "Precision", "Recall"]
    cols = st.columns(len(keys))
    colors_grad = ["#4CC9F0","#4361EE","#7209B7","#F72585","#B5179E"]
    for col, key, grad in zip(cols, keys, colors_grad):
        val = gm.get(key, "N/A")
        display = f"{val:.4f}" if isinstance(val, float) else str(val)
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{key.replace("_"," ")}</div>
                <div class="value" style="background:linear-gradient(135deg,{grad},{grad}99);
                    -webkit-background-clip:text;">{display}</div>
                </div>""", unsafe_allow_html=True)

    if "Threshold" in gm:
        st.markdown(f"""<div style='margin-top:12px; padding:10px 16px; background:#12121F;
            border:1px solid rgba(76,201,240,0.3); border-radius:10px; font-size:13px;'>
            ⚙️ <b>Optimal Threshold (max-F1 via PR Curve):</b>
            <code style='color:#4CC9F0'>{gm['Threshold']}</code>
            &nbsp;·&nbsp; Applied instead of fixed 0.5 cutoff for better Recall/F1 balance.
        </div>""", unsafe_allow_html=True)

    # Confusion matrix
    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_bar = st.columns([1, 2])
    with col_cm:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = gm["Confusion_Matrix"]
        tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
        matrix = np.array([[tn, fp], [fn, tp]])
        fig, ax = dark_fig()
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred: No Prog", "Pred: Prog"], color="white", fontsize=9)
        ax.set_yticklabels(["Actual: No Prog", "Actual: Prog"], color="white", fontsize=9)
        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white", fontsize=18, fontweight="bold")
        ax.set_title("Confusion Matrix", color="white", fontsize=11)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_bar:
        st.markdown('<div class="section-header">Subgroup ROC-AUC by Cancer Type</div>',
                    unsafe_allow_html=True)
        cancer_types = list(sg.keys())
        aucs   = [sg[c]["ROC_AUC"]  for c in cancer_types]
        f1s    = [sg[c]["F1_Score"]  for c in cancer_types]

        fig, ax = dark_fig()
        x = np.arange(len(cancer_types))
        w = 0.35
        bars1 = ax.bar(x - w/2, aucs, w, label="ROC-AUC", color="#4361EE", alpha=0.9)
        bars2 = ax.bar(x + w/2, f1s,  w, label="F1-Score",  color="#F72585", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(cancer_types, color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.8, color="white", linestyle="--", linewidth=0.8, alpha=0.3)
        ax.legend(facecolor="#1A1A2E", edgecolor="none", labelcolor="white")
        ax.set_title("Per-Cancer Type Metrics", color="white", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Subgroup table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Detailed Subgroup Metrics</div>', unsafe_allow_html=True)
    sg_rows = [{"Cancer Type": c, "Samples": v["sample_count"],
                "ROC-AUC": v["ROC_AUC"], "F1-Score": v["F1_Score"]}
               for c, v in sg.items()]
    df_sg = pd.DataFrame(sg_rows)
    st.dataframe(df_sg.style.background_gradient(subset=["ROC-AUC","F1-Score"], cmap="Blues"),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE INFERENCE (Manual Input Form)
# ══════════════════════════════════════════════════════════════════════════════
elif "Inference" in page:
    # ─── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero" style="padding:32px 36px; margin-bottom:24px;">
        <h1 style="font-size:28px; margin-bottom:8px;">🧬 Patient Progression Predictor</h1>
        <p style="font-size:14px; color:#8A8AB0;">
            Enter your patient's biomarker measurements across <b style="color:#4CC9F0;">5 timepoints (M0 → M3 → M6 → M9 → M12)</b>
            and click <b style="color:#4CC9F0;">Run Prediction</b> to get a real-time cancer progression risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not CKPT_PATH.exists():
        st.error(f"Model checkpoint not found. Run `03_deep_learning_pipeline.py` first.")
        st.stop()
    if not CSV_PATH.exists():
        st.error(f"Dataset CSV not found: `{CSV_PATH}`")
        st.stop()

    # ─── Load model (cached) ───────────────────────────────────────────────────
    @st.cache_resource(show_spinner="⚙️ Loading AI model…")
    def load_pipeline():
        sys.path.insert(0, str(STAGE2_DIR))
        spec = importlib.util.spec_from_file_location(
            "pipeline_mod", str(STAGE2_DIR / "05_pipeline_integration.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from sklearn.preprocessing import StandardScaler
        df_ref = pd.read_csv(CSV_PATH)
        scaler = StandardScaler()
        scaler.fit(df_ref[["ctDNA_vaf_pct","ct_tumor_vol_cm3","biomarker_ng_ml","wsi_atypia_score"]].values)
        pipe = mod.Stage02InferencePipeline(model_path=str(CKPT_PATH), scaler=scaler)
        return pipe, df_ref

    try:
        pipeline, df_ref = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # ─── Reference ranges (from dataset) ──────────────────────────────────────
    ref_min = df_ref[["ctDNA_vaf_pct","ct_tumor_vol_cm3","biomarker_ng_ml","wsi_atypia_score"]].min()
    ref_max = df_ref[["ctDNA_vaf_pct","ct_tumor_vol_cm3","biomarker_ng_ml","wsi_atypia_score"]].max()
    ref_mean = df_ref[["ctDNA_vaf_pct","ct_tumor_vol_cm3","biomarker_ng_ml","wsi_atypia_score"]].mean()

    # ─── Patient meta inputs ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)
    pi_c1, pi_c2, pi_c3 = st.columns([1, 1, 1])
    with pi_c1:
        patient_id_input = st.text_input("Patient ID", value="PAT-NEW-001", help="Alphanumeric patient identifier")
    with pi_c2:
        cancer_type_input = st.selectbox(
            "Cancer Type",
            ["BRCA","LUAD","LUSC","COAD","PRAD","STAD","SKCM","BLCA"],
            index=0
        )
    with pi_c3:
        st.markdown("<br>", unsafe_allow_html=True)
        load_example = st.button("📂 Load Example Patient", use_container_width=True,
                                  help="Auto-fill with a real patient from the dataset")

    # ─── Auto-fill with example patient ───────────────────────────────────────
    example_rows = None
    if load_example:
        sample_pat = df_ref[df_ref["cancer_type"] == cancer_type_input]["patient_id"].iloc[0]
        example_rows = df_ref[df_ref["patient_id"] == sample_pat].sort_values("sequence_step").reset_index(drop=True)
        st.success(f"Loaded example: **{sample_pat}** ({cancer_type_input})")

    # ─── Biomarker reference card ──────────────────────────────────────────────
    with st.expander("📊 Reference Ranges (Dataset Statistics)", expanded=False):
        cols_ref = st.columns(4)
        labels_ref = ["ctDNA VAF (%)", "CT Tumor Vol (cm³)", "Biomarker (ng/mL)", "WSI Atypia Score"]
        keys_ref   = ["ctDNA_vaf_pct","ct_tumor_vol_cm3","biomarker_ng_ml","wsi_atypia_score"]
        for col, label, key in zip(cols_ref, labels_ref, keys_ref):
            with col:
                st.markdown(f"""
                <div style="background:#1A1A2E; border:1px solid rgba(255,255,255,0.07);
                            border-radius:10px; padding:12px; text-align:center;">
                    <div style="font-size:11px; color:#8A8AB0; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:6px;">{label}</div>
                    <div style="color:#4CC9F0; font-weight:600;">Mean: {ref_mean[key]:.2f}</div>
                    <div style="color:#8A8AB0; font-size:11px;">
                        Range: {ref_min[key]:.1f} – {ref_max[key]:.1f}
                    </div>
                </div>""", unsafe_allow_html=True)

    # ─── 5-Timepoint Input Form ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Biomarker Data — 5 Timepoints</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#12121F; border:1px solid rgba(76,201,240,0.2); border-radius:12px;
                padding:14px 18px; margin-bottom:20px; font-size:13px; color:#8A8AB0;">
        ℹ️ Enter measurements at each monitoring timepoint.
        Values outside normal range are highlighted.
        All fields use real clinical units from the oncology dataset.
    </div>""", unsafe_allow_html=True)

    TIMEPOINTS   = [0, 3, 6, 9, 12]
    TP_LABELS    = ["M0 (Baseline)", "M3 (3 months)", "M6 (6 months)", "M9 (9 months)", "M12 (12 months)"]
    TP_ICONS     = ["🟢", "🔵", "🟡", "🟠", "🔴"]

    tabular_defaults = []
    if example_rows is not None:
        for i in range(5):
            row = example_rows.iloc[i]
            tabular_defaults.append({
                "ctDNA_vaf_pct":     float(row["ctDNA_vaf_pct"]),
                "ct_tumor_vol_cm3":  float(row["ct_tumor_vol_cm3"]),
                "biomarker_ng_ml":   float(row["biomarker_ng_ml"]),
                "wsi_atypia_score":  float(row["wsi_atypia_score"]),
            })
    else:
        for _ in range(5):
            tabular_defaults.append({
                "ctDNA_vaf_pct":     round(float(ref_mean["ctDNA_vaf_pct"]), 2),
                "ct_tumor_vol_cm3":  round(float(ref_mean["ct_tumor_vol_cm3"]), 2),
                "biomarker_ng_ml":   round(float(ref_mean["biomarker_ng_ml"]), 2),
                "wsi_atypia_score":  round(float(ref_mean["wsi_atypia_score"]), 2),
            })

    user_inputs = []
    tabs = st.tabs([f"{TP_ICONS[i]} {TP_LABELS[i]}" for i in range(5)])
    for i, tab in enumerate(tabs):
        with tab:
            d = tabular_defaults[i]
            st.markdown(f"""
            <div style="background:rgba(67,97,238,0.06); border-radius:10px; padding:12px 16px; margin-bottom:16px;">
                <b style="color:#4CC9F0;">{TP_ICONS[i]} Timepoint: Month {TIMEPOINTS[i]}</b>
                &nbsp;—&nbsp; <span style="color:#8A8AB0; font-size:13px;">Enter measurements taken at this visit</span>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                ctdna = st.number_input(
                    "ctDNA VAF (%)",
                    min_value=0.0, max_value=100.0,
                    value=d["ctDNA_vaf_pct"], step=0.1, format="%.2f",
                    key=f"ctdna_{i}",
                    help="Circulating tumor DNA variant allele frequency"
                )
            with c2:
                tumor_vol = st.number_input(
                    "CT Tumor Vol (cm³)",
                    min_value=0.0, max_value=5000.0,
                    value=d["ct_tumor_vol_cm3"], step=1.0, format="%.1f",
                    key=f"tvol_{i}",
                    help="Tumor volume measured from CT scan"
                )
            with c3:
                biomarker = st.number_input(
                    "Biomarker (ng/mL)",
                    min_value=0.0, max_value=1000.0,
                    value=d["biomarker_ng_ml"], step=0.5, format="%.2f",
                    key=f"bio_{i}",
                    help="Cancer-specific serum biomarker concentration"
                )
            with c4:
                atypia = st.number_input(
                    "WSI Atypia Score",
                    min_value=0.0, max_value=10.0,
                    value=d["wsi_atypia_score"], step=0.1, format="%.2f",
                    key=f"aty_{i}",
                    help="Whole-slide image atypia score (pathology)"
                )

            user_inputs.append({
                "ctDNA_vaf_pct":     ctdna,
                "ct_tumor_vol_cm3":  tumor_vol,
                "biomarker_ng_ml":   biomarker,
                "wsi_atypia_score":  atypia,
            })

            # Mini alert if values are high
            warnings = []
            if ctdna > ref_max["ctDNA_vaf_pct"] * 0.8:
                warnings.append("⚠️ ctDNA VAF is elevated")
            if tumor_vol > ref_max["ct_tumor_vol_cm3"] * 0.8:
                warnings.append("⚠️ Tumor volume is elevated")
            if atypia > 7.0:
                warnings.append("⚠️ High atypia score detected")
            if warnings:
                st.warning("  ·  ".join(warnings))

    # ─── Run Prediction Button ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        run_pred = st.button(
            "🚀  Run Progression Prediction",
            use_container_width=True,
            type="primary",
        )

    # ─── Prediction Results ────────────────────────────────────────────────────
    if run_pred:
        # Build a synthetic patient DataFrame compatible with the inference pipeline
        import uuid, datetime
        fake_img = "placeholder.png"  # pipeline has fallback for missing images
        rows = []
        for i, vals in enumerate(user_inputs):
            rows.append({
                "patient_id":        patient_id_input,
                "cancer_type":       cancer_type_input,
                "sequence_step":     i,
                "timepoint_month":   TIMEPOINTS[i],
                "ctDNA_vaf_pct":     vals["ctDNA_vaf_pct"],
                "ct_tumor_vol_cm3":  vals["ct_tumor_vol_cm3"],
                "biomarker_ng_ml":   vals["biomarker_ng_ml"],
                "wsi_atypia_score":  vals["wsi_atypia_score"],
                "pathology_img_file": fake_img,
                "ct_img_file":        fake_img,
                "label_progression":  0,  # dummy label for inference
            })
        patient_df = pd.DataFrame(rows)

        with st.spinner("🧠 Running multi-modal AI inference…"):
            try:
                result = pipeline.predict_patient_trajectory(patient_df, str(IMG_DIR))
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        final_prob   = result["trajectory_final_progression_probability"]
        overall_tier = result["overall_clinical_risk_tier"]
        tier_color   = {
            "Low Progression Risk":  "#4ade80",
            "Moderate Risk":         "#fbbf24",
            "High Progression Risk": "#f87171",
        }.get(overall_tier, "#4ade80")

        st.markdown("---")
        st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

        # ── Overall result card ──────────────────────────────────────────────
        res_c1, res_c2, res_c3 = st.columns([1, 1, 1])
        with res_c1:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{tier_color}55; border-width:2px;">
                <div class="label">Patient ID</div>
                <div class="value" style="font-size:20px; background:linear-gradient(135deg,#fff,#4CC9F0);
                    -webkit-background-clip:text;">{patient_id_input}</div>
                <div class="sub">{cancer_type_input}</div>
            </div>""", unsafe_allow_html=True)
        with res_c2:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{tier_color}55; border-width:2px;">
                <div class="label">Final Progression Probability</div>
                <div class="value" style="background:linear-gradient(135deg,{tier_color},{tier_color}88);
                    -webkit-background-clip:text;">{final_prob:.1%}</div>
                <div class="sub">At Month 12</div>
            </div>""", unsafe_allow_html=True)
        with res_c3:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{tier_color}55; border-width:2px;">
                <div class="label">Clinical Risk Tier</div>
                <div style="margin-top:12px; font-size:20px;">{risk_badge(overall_tier)}</div>
                <div class="sub" style="margin-top:10px;">AI-assessed risk classification</div>
            </div>""", unsafe_allow_html=True)

        # ── Trajectory chart ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Progression Trajectory</div>', unsafe_allow_html=True)

        tps    = result["timepoint_predictions"]
        steps  = [t["step"] for t in tps]
        months = [f"M{t['timepoint_month']}" for t in tps]
        probs  = [t["progression_probability"] for t in tps]

        fig, ax = dark_fig()
        fig.set_size_inches(10, 4)

        # Gradient-like fill
        ax.fill_between(steps, probs, alpha=0.12, color="#4CC9F0")

        # Color-coded dots by risk
        dot_colors = []
        for p in probs:
            if p < 0.35:
                dot_colors.append("#4ade80")
            elif p <= 0.70:
                dot_colors.append("#fbbf24")
            else:
                dot_colors.append("#f87171")

        ax.plot(steps, probs, color="#4CC9F0", linewidth=2.5, zorder=3)
        for s, p, dc in zip(steps, probs, dot_colors):
            ax.scatter(s, p, color=dc, s=120, zorder=4, edgecolors="white", linewidth=1.5)
            ax.annotate(f"{p:.3f}", (s, p),
                        textcoords="offset points", xytext=(0, 14),
                        ha="center", color="white", fontsize=10, fontweight="bold")

        # Risk zones
        ax.axhspan(0,    0.35, alpha=0.04, color="#4ade80")
        ax.axhspan(0.35, 0.70, alpha=0.04, color="#fbbf24")
        ax.axhspan(0.70, 1.05, alpha=0.04, color="#f87171")
        ax.axhline(0.35, color="#fbbf24", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(0.70, color="#f87171", linestyle="--", linewidth=1, alpha=0.5)

        ax.text(4.6, 0.18, "LOW", color="#4ade80", fontsize=9, ha="right", alpha=0.7)
        ax.text(4.6, 0.52, "MODERATE", color="#fbbf24", fontsize=9, ha="right", alpha=0.7)
        ax.text(4.6, 0.87, "HIGH", color="#f87171", fontsize=9, ha="right", alpha=0.7)

        ax.set_xticks(steps)
        ax.set_xticklabels(months, color="white", fontsize=11)
        ax.set_ylabel("Progression Probability", color="white", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
        ax.set_title(f"Cancer Progression Trajectory — {patient_id_input} ({cancer_type_input})",
                     color="white", fontsize=12, fontweight="bold", pad=14)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Per-timepoint breakdown table ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Timepoint Breakdown</div>', unsafe_allow_html=True)

        tp_rows = []
        for t, inp in zip(tps, user_inputs):
            badge = risk_badge(t["risk_tier"])
            prob_bar_pct = int(t["progression_probability"] * 100)
            bar_color = {"Low Progression Risk":"#4ade80","Moderate Risk":"#fbbf24",
                         "High Progression Risk":"#f87171"}.get(t["risk_tier"],"#4ade80")
            bar_html = (f'<div style="width:100%; background:#1A1A2E; border-radius:4px; height:8px;">'
                        f'<div style="width:{prob_bar_pct}%; background:{bar_color}; height:8px; border-radius:4px;"></div>'
                        f'</div><small style="color:#8A8AB0">{prob_bar_pct}%</small>')
            tp_rows.append(
                f"<tr>"
                f"<td><b>M{t['timepoint_month']}</b></td>"
                f"<td>{inp['ctDNA_vaf_pct']:.2f}%</td>"
                f"<td>{inp['ct_tumor_vol_cm3']:.1f}</td>"
                f"<td>{inp['biomarker_ng_ml']:.2f}</td>"
                f"<td>{inp['wsi_atypia_score']:.2f}</td>"
                f"<td>{bar_html}</td>"
                f"<td>{badge}</td>"
                f"</tr>"
            )

        html_table = f"""
        <table class="styled-table">
            <thead><tr>
                <th>Timepoint</th>
                <th>ctDNA VAF (%)</th>
                <th>Tumor Vol (cm³)</th>
                <th>Biomarker (ng/mL)</th>
                <th>WSI Atypia</th>
                <th>Progression Prob.</th>
                <th>Risk Tier</th>
            </tr></thead>
            <tbody>{''.join(tp_rows)}</tbody>
        </table>"""
        st.markdown(html_table, unsafe_allow_html=True)

        # ── Clinical recommendation ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        rec_text = {
            "Low Progression Risk":  ("✅ Low Risk", "#4ade80",
                "Continue routine monitoring. No immediate intervention required. "
                "Schedule next assessment in 6 months."),
            "Moderate Risk":         ("⚠️ Moderate Risk", "#fbbf24",
                "Increased surveillance recommended. Consider adjuvant therapy consultation. "
                "Schedule follow-up imaging in 3 months."),
            "High Progression Risk": ("🔴 High Risk — Immediate Action Required", "#f87171",
                "Patient shows high probability of cancer progression. "
                "Urgent oncology consultation recommended. "
                "Consider treatment escalation and multidisciplinary team review."),
        }.get(overall_tier, ("✅ Low Risk", "#4ade80", "Routine monitoring advised."))

        st.markdown(f"""
        <div style="background:#12121F; border:2px solid {rec_text[1]}33;
                    border-left:4px solid {rec_text[1]}; border-radius:12px;
                    padding:20px 24px; margin-top:8px;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:16px;
                        font-weight:700; color:{rec_text[1]}; margin-bottom:10px;">
                {rec_text[0]}
            </div>
            <div style="color:#C0C0E0; font-size:14px; line-height:1.7;">
                {rec_text[2]}
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        # Show prompt to run prediction
        st.markdown("""
        <div style="text-align:center; padding:48px 24px; background:#12121F;
                    border:1px dashed rgba(76,201,240,0.3); border-radius:16px; margin-top:24px;">
            <div style="font-size:56px; margin-bottom:16px;">🚀</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:20px;
                        font-weight:600; color:#F0F0FF; margin-bottom:10px;">
                Ready to Predict
            </div>
            <div style="color:#8A8AB0; font-size:14px; max-width:480px; margin:0 auto; line-height:1.7;">
                Fill in the patient biomarker data for all 5 timepoints above,
                then click <b style="color:#4CC9F0;">Run Progression Prediction</b> to get
                the AI-powered cancer progression risk assessment.
            </div>
        </div>""", unsafe_allow_html=True)

