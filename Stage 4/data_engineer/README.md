# Stage 04 Oncology SLM Data Engineering (ETL Pipeline)

## Overview & Architecture
This module executes the **Stage 3 to Stage 04 Data Transformation Pipeline** (`stage04_etl_pipeline.py`). It ingests processed oncology datasets from Stage 3 and transforms them into a high-precision, zero-leakage instruction dataset (`stage04_slm_train.jsonl`) specifically formatted for fine-tuning a 3B Small Language Model (SLM) to generate tumor-board clinical briefings.

## Input Datasets (Stage 3 Processed)
Read from `Stage 3/data_engineer/data/processed/`:
1. `urgency_dataset.csv` (10,000 records)
2. `ner_dataset.jsonl` (10,000 records)
3. `drug_knowledge.csv` (1,000 records)
4. `gene_mutation_dictionary.csv` (16 records)
5. `guideline_chunks.csv` (8 records)

## Transformation Rules & Constraints
- **Strict 2-Sentence Output Constraint:**
  - **Sentence 1:** Patient ID, cancer subtype, biomarker, drug regimen, triage urgency tier, and presentation symptoms.
  - **Sentence 2:** Immediate action recommendation mapped to triage level and clinical SOP guidelines.
- **Zero Data Leakage:** 0% contamination rate between train ($N=8,000$) and test ($N=2,000$) splits.
- **Deterministic Alignment:** Exact patient ID alignment from `P00001` through `P10000`.

## Schema
```json
{
  "instruction": "Summarize the extracted Stage 3 oncology NLP data into an actionable, 2-sentence clinical briefing for a tumor board.",
  "input": "PATIENT_ID: P00042\nDIAGNOSIS: Non-Small Cell Lung Cancer\nBIOMARKER: EGFR L858R\nREGIMEN: Osimertinib\nTRIAGE TIER: HIGH\nCLINICAL NOTE: Persistent intractable vomiting (>5x/day, dehydration), spiking fever of 38.6C with rigors during neutropenic nadir.",
  "output": "Patient P00042 (Non-Small Cell Lung Cancer, EGFR L858R) on Osimertinib presents with HIGH-tier urgency symptoms detailed as persistent intractable vomiting and spiking fever of 38.6C with rigors during neutropenic nadir. Recommend immediate clinical review and triage management according to HIGH protocol guidelines."
}
```

## How to Run
From the repository root:
```bash
python "Stage 4/data_engineer/stage04_etl_pipeline.py"
```

## Generated Artifacts
- `data/processed/stage04_slm_train.jsonl` (10,000 instruction-tuning records, ~7.58 MB)
- `outputs/stage04_data_quality_report.json` (Full quality & zero-leakage audit report)
