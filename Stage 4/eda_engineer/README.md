# Stage 04 EDA Engineer

## Overview
The **Stage 04 EDA Engineer** sub-pipeline executes clinical exploratory data analysis (EDA), statistical length distribution profiling, class balance verification, and clinical fidelity auditing on the Stage 04 SLM instruction dataset (`stage04_slm_train.jsonl`).

---

## Directory Structure
```
Stage 4/eda_engineer/
├── data/
│   └── processed/
│       ├── drug_knowledge.csv
│       ├── gene_mutation_dictionary.csv
│       ├── guideline_chunks.csv
│       └── stage04_slm_train.jsonl
├── outputs/
│   └── stage04_eda_report.json
├── src/
│   └── stage04_eda_audit.py
├── stage04_eda_pipeline.py
└── README.md
```

---

## Execution
Run from project root:
```powershell
python "Stage 4/eda_engineer/stage04_eda_pipeline.py"
```
Or directly:
```powershell
python "Stage 4/eda_engineer/src/stage04_eda_audit.py"
```

---

## Audit Standards & Measured Performance

| Dimension | Constraint / Threshold | Measured Result | Audit Status |
| :--- | :--- | :---: | :---: |
| **Total Records** | Exactly 10,000 | **10,000** | **PASSED** |
| **Schema Completeness** | `instruction`, `input`, `output` present | **100.0% (0 missing)** | **PASSED** |
| **2-Sentence Compliance** | Exactly 2 sentences per `output` | **100.0% (0 violations)** | **PASSED** |
| **Urgency Class Balance** | Entropy Ratio $\ge 0.95$ | **0.9855 (1.9710 bits)** | **PASSED** |
| **Biomarker Coverage** | 16 catalog biomarkers represented | **16 / 16 (100%)** | **PASSED** |
| **Drug Class Coverage** | 26 catalog drugs represented | **26 / 26 (100%)** | **PASSED** |
| **Entity Fallback Nulls** | No orphaned `None`/`NaN`/`Unknown` | **0 Null Entities** | **PASSED** |
| **Patient Duplication** | 1 unique ID per record | **10,000 Unique IDs (0.0% dup)** | **PASSED** |
| **Prompt-Target Leakage** | 0% Contamination | **0.0% Contamination** | **PASSED** |
