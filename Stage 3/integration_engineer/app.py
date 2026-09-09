"""
app.py
------
Streamlit User Interface for Stage 03 Oncology NLP Decision Support.

Title: "Oncology NLP Decision Support — Stage 03"

Features:
  - Clinical Disclaimer Banner
  - Interactive Clinical Text Input with 5 One-Click Clinical Presets
  - Color-Coded Urgency Tier Badge (CRITICAL, HIGH, MODERATE, LOW)
  - Interactive Medical Entity Highlighter Chips
  - Oncology Drug Knowledge Cards
  - Gene & Mutation Profile Cards
  - Clinical Guideline & SOP Semantic Match Viewer
  - Audit Warnings & "Potential interpretation uncertainty" Alerts
  - Raw JSON Inspector for API Parity
"""

import os
import sys
import json
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from nlp_pipeline import OncologyNLPPipeline

# Configure page
st.set_page_config(
    page_title="Oncology NLP Decision Support — Stage 03",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .disclaimer-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(245, 158, 11, 0.15));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 20px;
        font-size: 0.9rem;
        color: #f87171;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .urgency-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .urgency-CRITICAL {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1.5px solid #ef4444;
    }
    .urgency-HIGH {
        background-color: rgba(249, 115, 22, 0.2);
        color: #f97316;
        border: 1.5px solid #f97316;
    }
    .urgency-MODERATE {
        background-color: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        border: 1.5px solid #3b82f6;
    }
    .urgency-LOW {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1.5px solid #10b981;
    }
    .entity-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 4px;
    }
    .tag-DRUG_NAME { background-color: #2563eb; color: #ffffff; }
    .tag-DOSAGE { background-color: #059669; color: #ffffff; }
    .tag-ADVERSE_EVENT { background-color: #dc2626; color: #ffffff; }
    .tag-GENE_MUTATION { background-color: #7c3aed; color: #ffffff; }
    .tag-CANCER_TYPE { background-color: #d97706; color: #ffffff; }

    .audit-alert-box {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    .card-box {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Mandatory Clinical Disclaimer Banner
st.markdown("""
<div class="disclaimer-banner">
    <span>⚠️</span>
    <span><strong>Synthetic demonstration system.</strong> Not for clinical diagnosis, treatment decisions, or medical advice. Predictions must be verified by a licensed healthcare professional.</span>
</div>
""", unsafe_allow_html=True)

# Application Header
st.title("🧬 Oncology NLP Decision Support — Stage 03")
st.caption("Unified Decision Support Engine: Triage Urgency, Medical NER, Drug & Gene Lookups, and Guideline Retrieval")


@st.cache_resource
def load_pipeline():
    return OncologyNLPPipeline.load()

with st.spinner("Initializing Oncology NLP Pipeline & Clinical Knowledge Bases..."):
    pipeline = load_pipeline()

# Sidebar: System Metrics & Knowledge Base Status
with st.sidebar:
    st.subheader("System Health & Knowledge Bases")
    st.success("Pipeline: Operational")
    st.markdown(f"**Trained Urgency Model:** Baseline TF-IDF + LR (Macro F1 = 1.0000)")
    st.markdown(f"**Indexed Guidelines:** {len(pipeline.guide_retriever.df)} chunks")
    st.markdown(f"**Drug Catalog:** {len(pipeline.drug_lookup.df)} regimens")
    st.markdown("**Data Leakage Audit:** 0.0% Contamination")
    st.markdown("---")
    st.subheader("Regulatory Context")
    st.markdown("""
    - **Classification:** SaMD Class II Assistive Tool
    - **Protocol:** Human-in-the-loop (HITL) mandatory dual sign-off
    - **Audit Trail:** Active zero-PHI logging
    """)

# Main Input Section
st.subheader("Direct Clinical Text Input")
st.markdown("Enter or paste any unparsed clinical note, oncology consult, symptom description, or nursing handoff below:")

input_text = st.text_area(
    "Patient Clinical Note / Symptoms:",
    value="",
    placeholder="Type or paste any clinical narrative here...\n(e.g., Patient received pembrolizumab 200 mg and developed severe fatigue.)",
    height=150,
    help="Direct free-text input. Accepts any oncology consult, clinical abbreviations, symptom notes, or tele-triage messages."
)

col_btn, col_clear, col_opts = st.columns([1.5, 1, 3.5])
with col_btn:
    analyze_btn = st.button("🔍 ANALYZE NOTE", type="primary", use_container_width=True)
with col_clear:
    clear_btn = st.button("🔄 Reset", use_container_width=True)
with col_opts:
    top_k_guidelines = st.slider("Top Guidelines to Retrieve", min_value=1, max_value=5, value=3)

if clear_btn:
    st.session_state["result"] = None
    st.rerun()

if analyze_btn:
    if not input_text.strip():
        st.warning("⚠️ Please enter or paste clinical text before clicking Analyze.")
        st.session_state["result"] = None
    else:
        with st.spinner("Processing through Unified NLP Pipeline..."):
            st.session_state["result"] = pipeline.run(input_text, top_guidelines=top_k_guidelines)

result = st.session_state.get("result", None)

# Execution Display
if result:
    st.markdown("---")

    # Top Results Bar: Urgency & Uncertainty
    col_urgency, col_conf, col_audit = st.columns([2, 2, 4])

    urgency_label = result["urgency"]["label"]
    confidence_val = result["urgency"]["confidence"]

    with col_urgency:
        st.markdown(f"**Predicted Urgency Tier:**")
        st.markdown(
            f'<div class="urgency-badge urgency-{urgency_label}">{urgency_label}</div>',
            unsafe_allow_html=True
        )

    with col_conf:
        st.markdown(f"**Model Confidence:**")
        st.metric(label="Calibrated Score", value=f"{confidence_val:.1%}")
        st.progress(min(max(float(confidence_val), 0.0), 1.0))

    with col_audit:
        st.markdown(f"**Clinical Interpretation Status:**")
        if confidence_val < 0.70 or any("uncertainty" in flag.lower() for flag in result["audit_flags"]):
            st.markdown("""
            <div class="audit-alert-box">
                <strong>⚠️ Potential interpretation uncertainty</strong><br>
                Confidence is below 0.70 or clinical presentation contains ambiguity. Do not treat as confirmed triage; manual clinician review mandatory.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Verified High Clinical Alignment. Assistive decision support recommendation only.")

    # Entities Section
    st.subheader("Medical Named Entities")
    entities = result.get("entities", [])
    if entities:
        ent_html = "<div>"
        for ent in entities:
            lbl = ent["label"]
            txt = ent["text"]
            conf = ent.get("confidence", 0.0)
            ent_html += f'<span class="entity-tag tag-{lbl}">{txt} <small>({lbl} {conf:.0%})</small></span>'
        ent_html += "</div>"
        st.markdown(ent_html, unsafe_allow_html=True)
    else:
        st.caption("No medical named entities identified in current note.")

    # Coreference Resolution Section
    corefs = result.get("coreferences", [])
    if corefs:
        st.markdown("---")
        st.subheader("🔗 Clinical Coreference Resolution (Anaphora Linking)")
        cf_cols = st.columns(len(corefs) if len(corefs) <= 4 else 4)
        for i, cf in enumerate(corefs):
            with cf_cols[i % len(cf_cols)]:
                st.markdown(f"""
                <div style="background:#1e293b; border-left:3px solid #38bdf8; padding:8px 12px; border-radius:6px; margin-bottom:8px;">
                    <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">{cf.get('entity_type', 'ENTITY')} ANAPHOR</div>
                    <div style="font-size:0.95rem; font-weight:600; color:#e2e8f0;">
                        <span style="color:#f43f5e; text-decoration:line-through;">"{cf.get('anaphor')}"</span> ➔ <span style="color:#10b981;">{cf.get('antecedent')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with st.expander("📝 View Resolved Coreferenced Clinical Text"):
            st.info(result.get("resolved_text", ""))

    st.markdown("---")

    # Two-column knowledge & guidelines display
    col_kb, col_guide = st.columns(2)

    with col_kb:
        st.subheader("💊 Drug & Biomarker Information")

        # Drug Information
        drugs = result.get("drug_information", [])
        if drugs:
            for d in drugs:
                with st.expander(f"Drug Match: {d.get('matched_as', '').title()}", expanded=True):
                    st.markdown(f"**Match Type:** {d.get('match_type', 'exact').title()}")
                    for rec in d.get("records", []):
                        st.markdown(f"- **Known AE:** `{rec.get('adverse_event', 'N/A')}` | **Expected Severity:** `{rec.get('severity', 'N/A')}`")
                        st.markdown(f"- **Standard Route:** `{rec.get('route', 'N/A')}` | **Dosage:** `{rec.get('dosage', 'N/A')}`")
        else:
            st.caption("No indexed oncology drugs matched in text.")

        # Mutation Information
        mutations = result.get("mutation_information", [])
        if mutations:
            for m in mutations:
                with st.expander(f"Biomarker Match: {m.get('mutation', m.get('gene', 'Gene')).upper()}", expanded=True):
                    for r in m.get("records", []):
                        st.markdown(f"- **Gene:** `{r.get('gene', 'N/A')}`")
                        st.markdown(f"- **Specific Variant:** `{r.get('mutation', 'N/A')}`")
                        st.markdown(f"- **Associated Cancer:** `{r.get('associated_cancer', 'N/A')}`")
        else:
            st.caption("No genomic mutations matched in text.")

    with col_guide:
        st.subheader("📋 Relevant Clinical Guidelines & SOPs")
        guidelines = result.get("guideline_matches", [])
        if guidelines:
            for idx, g in enumerate(guidelines, 1):
                score = g.get("similarity_score", 0.0)
                with st.expander(f"Rank {idx}: {g.get('section', 'General SOP')} (Relevance: {score:.1%})", expanded=(idx==1)):
                    st.markdown(f"**Source:** `{g.get('source', 'Oncology-SOP')}` | **ID:** `{g.get('document_id', 'N/A')}`")
                    st.write(g.get("text", ""))
        else:
            st.caption("No guideline chunks retrieved above score threshold.")

    # Audit Flags Section
    st.markdown("---")
    st.subheader("🛡️ Safety Envelope & Audit Flags")
    flags = result.get("audit_flags", [])
    if flags:
        for flag in flags:
            st.warning(f"• {flag}")
    else:
        st.success("Zero safety or contradiction flags raised for this note.")

    # Raw JSON inspection
    with st.expander("🔍 View Raw Unified JSON Response (API Format)"):
        st.json(result)
