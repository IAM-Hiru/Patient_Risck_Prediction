"""
stage04_eval_pipeline.py
------------------------
Main runner for Stage 04 Evaluation Engineer Pipeline.
Executes test evaluation on 500 held-out samples, computes NLP metrics,
generates confusion matrix and benchmark visualizations.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from stage04_evaluate_slm import run_evaluation
from generate_eval_charts import generate_evaluation_visualizations
from stage04_hallucination_audit import run_clinical_safety_audit
from stage04_test_endpoint import run_endpoint_verification

if __name__ == "__main__":
    _STAGE4_DIR = os.path.dirname(_HERE)
    dataset_path = os.path.join(_STAGE4_DIR, "data_engineer", "data", "processed", "stage04_slm_train.jsonl")
    adapter_path = os.path.join(_STAGE4_DIR, "slm_engineer", "models", "stage04_slm_qlora")
    out_dir = os.path.join(_HERE, "outputs")
    report_path = os.path.join(out_dir, "stage04_evaluation_report.json")
    safety_report_path = os.path.join(out_dir, "stage04_clinical_safety_report.json")
    endpoint_audit_path = os.path.join(out_dir, "stage04_endpoint_audit.json")

    # 1. Run Quantitative Evaluation (if needed or verify report exists)
    if not os.path.exists(report_path):
        run_evaluation(
            dataset_path=dataset_path,
            adapter_path=adapter_path,
            output_report_path=report_path,
            test_samples_count=500
        )

    # 2. Generate Visual Charts
    generate_evaluation_visualizations(
        report_path=report_path,
        output_dir=out_dir
    )

    # 3. Run Clinical Safety & Hallucination Audit
    run_clinical_safety_audit(
        predictions_path=os.path.join(out_dir, "stage04_test_predictions.jsonl"),
        output_report_path=safety_report_path
    )

    # 4. Run FastAPI Microservice Endpoint Verification Suite
    run_endpoint_verification(
        audit_output_path=endpoint_audit_path
    )

