# Oncology NLP Decision Support System — Stage 03

> **CLINICAL DISCLAIMER & SAFETY NOTICE**  
> **Synthetic demonstration system. Not for clinical diagnosis, treatment decisions, or medical advice.**  
> All outputs must be validated by licensed healthcare professionals. Do not deploy for autonomous clinical decision-making.

---

## 1. Overview & Architecture

The **Stage 03 Oncology NLP Decision Support System** unifies all machine learning models, rule-based clinical parsers, and curated oncology reference databases into a production-ready, locally runnable application.

### Integrated Pipeline Flow

```
Clinical Text / Consult Note
        │
        ▼
[ 1. Preprocessing & Abbreviation Expansion ]
  - Oncology abbreviation dictionary (SOB, n/v, HA, neutropenic, etc.)
  - Text normalization, spacing, and sentence segmentation
        │
        ▼
[ 2. Hybrid Clinical NER ]
  - Recognizes DRUG_NAME, DOSAGE, ADVERSE_EVENT, GENE_MUTATION
  - Compound symptom & dosage matching (e.g., "175 mg/m2", "severe fatigue")
        │
        ▼
[ 3. Clinical Negation & Polarity Filtering ]
  - Negation cue detection ("denies", "no", "negative for", "hx of")
  - Distinguishes active acute symptoms from denied/historical entities
        │
        ▼
[ 4. Triage Urgency Classification ]
  - TF-IDF + Logistic Regression calibrated probability distribution
  - Clinical safety override rules (Critical, High, Moderate, Low)
  - Negative biomarker and pure negation handling
        │
        ▼
[ 5. Knowledge Base & SOP Retrieval ]
  - Drug Knowledge Base lookup (mechanism, indications, adverse events, warnings)
  - Gene & Mutation Dictionary lookup (EGFR L858R, KRAS G12C, BRAF V600E, etc.)
  - Clinical Guideline Retriever (BM25 + TF-IDF cosine similarity over oncology SOPs)
        │
        ▼
[ 6. Evaluation, Uncertainty & Safety Auditing ]
  - Confidence threshold monitoring (< 0.65 flags review)
  - Cross-checking missing drugs/biomarkers, conflicting findings
  - Privacy-preserving audit logging (zero PHI retained)
        │
        ├──▶ Streamlit Clinical Dashboard (app.py)
        └──▶ FastAPI REST Microservice (api.py)
```

---

## 2. Directory Structure

```
Stage 3/integration_engineer/
├── api.py                     # FastAPI service (/health, /analyze)
├── app.py                     # Streamlit clinical interactive dashboard
├── requirements.txt           # Python dependency specifications
├── README.md                  # System documentation & usage guide
├── data/
│   └── processed/
│       ├── drug_knowledge.csv          # Curated oncology drug reference database
│       ├── gene_mutation_dictionary.csv # Biomarker & variant mapping
│       ├── guideline_chunks.csv        # Clinical oncology SOPs & management guidelines
│       └── urgency_dataset.csv         # Training/validation clinical triage corpus
├── models/
│   ├── urgency/
│   │   └── baseline_tfidf_lr.pkl       # Serialized TF-IDF + Logistic Regression triage model
│   └── ner/
│       └── ml_token_ner.pkl            # Serialized ML token NER model
├── src/
│   ├── __init__.py
│   ├── nlp_pipeline.py         # Unified OncologyNLPPipeline orchestrator
│   ├── urgency_classifier.py   # Triage classifier wrapper
│   ├── ner_model.py            # Rule-based & compound NER extractors
│   ├── preprocessing.py        # Abbreviation normalizer & tokenizer
│   ├── drug_lookup.py          # Drug reference profile matcher
│   ├── gene_lookup.py          # Biomarker variant lookup engine
│   ├── guideline_retrieval.py  # Guideline retrieval engine (TF-IDF + Cosine)
│   └── logger.py               # HIPAA-conscious de-identified audit logger
├── logs/
│   └── integration_audit.log   # Append-only audit log (metadata only, no raw patient text)
├── outputs/                    # Exported analysis results and cached reports
└── tests/
    ├── test_pipeline.py        # Core pipeline unit and integration tests
    ├── test_api.py             # FastAPI endpoint contract & error handling tests
    └── test_edge_cases.py      # Edge case, abbreviation, and negation tests
```

---

## 3. Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Windows / macOS / Linux
- **No Docker required**
- **No external paid/cloud LLM APIs required** (100% local execution)

### 1. Install Dependencies
In your terminal or PowerShell:
```powershell
cd "Stage 3/integration_engineer"
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
Run the comprehensive 16-test suite:
```powershell
python -m pytest tests/ -v
```

### 3. Launch Streamlit Web Application
```powershell
streamlit run app.py
```
- Opens automatically in your browser at `http://localhost:8501`.
- Features 6 one-click oncology clinical preset scenarios:
  1. *Pembrolizumab Immunotherapy Toxicity* (Severe Fatigue)
  2. *Severe Osimertinib Pulmonary Distress* (Critical Shortness of Breath)
  3. *Paclitaxel / Carboplatin Combination Regimen* (Complex Multi-Drug)
  4. *Negative Biomarker Screening* (EGFR Negation)
  5. *Explicit Multi-Symptom Negation* (Denies Symptoms)
  6. *Heavy Clinical Shorthand / Abbreviations* (`Pt hx of SOB, c/o severe n/v...`)

### 4. Launch FastAPI Microservice
```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

---

## 4. API Specification & Schema Contract

### `GET /health`
Returns system status, loaded model components, and the clinical disclaimer.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "Oncology NLP Decision Support API",
  "version": "3.0.0",
  "disclaimer": "Synthetic demonstration system. Not for clinical diagnosis, treatment decisions, or emergency patient triage.",
  "timestamp": "2026-09-09T05:30:00.000000+00:00",
  "components": {
    "urgency_classifier": "loaded (classes: 4)",
    "ner_extractor": "loaded",
    "drug_knowledge_base": "loaded (12 drugs)",
    "gene_mutation_dictionary": "loaded (28 mutations)",
    "guideline_retriever": "loaded (8 guideline chunks)"
  }
}
```

---

### `POST /analyze`
Analyzes a clinical note or oncology consult text.

**Request Payload:**
```json
{
  "text": "Patient received pembrolizumab 200 mg and developed severe fatigue.",
  "top_guidelines": 2
}
```

**Response Payload (200 OK):**
```json
{
  "text": "Patient received pembrolizumab 200 mg and developed severe fatigue.",
  "urgency": {
    "label": "HIGH",
    "confidence": 0.8814
  },
  "entities": [
    {
      "text": "pembrolizumab",
      "label": "DRUG_NAME",
      "confidence": 0.95,
      "negated": false
    },
    {
      "text": "200 mg",
      "label": "DOSAGE",
      "confidence": 0.9,
      "negated": false
    },
    {
      "text": "severe fatigue",
      "label": "ADVERSE_EVENT",
      "confidence": 0.92,
      "negated": false
    }
  ],
  "drug_information": [
    {
      "found": true,
      "matched_as": "pembrolizumab",
      "class": "PD-1 Inhibitor",
      "mechanism_of_action": "Blocks PD-1 receptor to restore antitumor T-cell activity.",
      "indications": "Melanoma, NSCLC, Head and Neck Squamous Cell Carcinoma, Urothelial Carcinoma",
      "adverse_events": [
        "Fatigue",
        "Pruritus",
        "Diarrhea",
        "Immune-mediated pneumonitis"
      ],
      "black_box_warning": "Immune-mediated adverse reactions affecting any organ system; infusion-related reactions."
    }
  ],
  "mutation_information": [],
  "guideline_matches": [
    {
      "guideline_id": "G001",
      "title": "Immune Checkpoint Inhibitor-Related Toxicity Management",
      "grade": "Grade 2-3 Toxicity",
      "action": "Withhold pembrolizumab. Initiate oral prednisone 1-2 mg/kg/day.",
      "monitoring": "Monitor liver function tests, thyroid panel, and symptom progression every 48 hours.",
      "escalation": "If symptoms worsen to Grade 3-4, admit for intravenous methylprednisolone.",
      "relevance_score": 0.3842
    }
  ],
  "audit_flags": [
    "CLINICAL SAFETY NOTICE: Urgency evaluated as HIGH. Prompt oncology triage recommended."
  ]
}
```

---

## 5. Security & Privacy-Preserving Audit Logging

To comply with HIPAA de-identification standards, patient clinical narrative text is **never** written to persistent disk logs.

All audit entries in `logs/integration_audit.log` record only execution metadata:
- ISO 8601 UTC timestamp
- Input character count (`input_text_length`)
- Predicted urgency label & confidence score
- Number of extracted clinical entities
- Execution latency in milliseconds
- Safety / audit warning flags

Example log record:
```json
{"timestamp": "2026-09-09T05:30:15.123456+00:00", "input_text_length": 68, "predicted_urgency": "HIGH", "urgency_confidence": 0.8814, "num_entities": 3, "latency_ms": 32.4, "audit_flags": ["CLINICAL SAFETY NOTICE: Urgency evaluated as HIGH. Prompt oncology triage recommended."]}
```

---

## 6. Verification & Test Suite Summary

The automated test suite covers unit logic, API schema constraints, and edge cases:
- **`test_pipeline.py`**: Validates pipeline loading, exact JSON contract structure, empty text fallbacks, and calibrated confidence bounds.
- **`test_api.py`**: Validates `/health` status, `/analyze` response schema parity, empty string validation (HTTP 400), whitespace handling (HTTP 400), and missing parameter validation (HTTP 422).
- **`test_edge_cases.py`**: Validates multi-drug combination protocols, multi-symptom acuity, explicit negations, clinical abbreviations (`SOB`, `n/v`, `HA`), unknown drug terminology handling, and negative biomarker results.

All 16 test cases pass with zero failures.
