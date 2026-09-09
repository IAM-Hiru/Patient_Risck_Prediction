"""
stage04_safety_pipeline.py
--------------------------
Main runner for Stage 04 Clinical Safety, Risk & Hallucination Audit Pipeline.
Performs zero-tolerance medical audit on 500 generated test outputs.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from stage04_hallucination_audit import run_clinical_safety_audit

if __name__ == "__main__":
    _STAGE4_DIR = os.path.dirname(os.path.dirname(_HERE))
    preds_path = os.path.join(_HERE, "outputs", "stage04_test_predictions.jsonl")
    drug_path = os.path.join(_STAGE4_DIR, "data_engineer", "data", "processed", "drug_knowledge.csv")
    gene_path = os.path.join(_STAGE4_DIR, "data_engineer", "data", "processed", "gene_mutation_dictionary.csv")
    out_report = os.path.join(_HERE, "outputs", "stage04_clinical_safety_report.json")

    run_clinical_safety_audit(
        predictions_path=preds_path,
        drug_csv_path=drug_path,
        gene_csv_path=gene_path,
        output_report_path=out_report
    )
