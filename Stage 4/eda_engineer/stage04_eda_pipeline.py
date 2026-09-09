"""
stage04_eda_pipeline.py
-----------------------
Main runner for Stage 04 EDA Engineer Pipeline.
Executes statistical profiling, format auditing, entropy analysis, and clinical fidelity checks.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from stage04_eda_audit import run_eda_audit
from generate_eda_charts import generate_eda_visualizations

if __name__ == "__main__":
    _STAGE4_DIR = os.path.dirname(_HERE)
    data_train_jsonl = os.path.join(_STAGE4_DIR, "data_engineer", "data", "processed", "stage04_slm_train.jsonl")
    data_gene_csv = os.path.join(_STAGE4_DIR, "data_engineer", "data", "processed", "gene_mutation_dictionary.csv")
    data_drug_csv = os.path.join(_STAGE4_DIR, "data_engineer", "data", "processed", "drug_knowledge.csv")
    out_dir = os.path.join(_HERE, "outputs")
    out_report_json = os.path.join(out_dir, "stage04_eda_report.json")

    # 1. Run Quantitative Audit
    run_eda_audit(
        dataset_path=data_train_jsonl,
        gene_dict_path=data_gene_csv,
        drug_dict_path=data_drug_csv,
        output_report_path=out_report_json
    )

    # 2. Generate Visual Charts & Dashboard
    generate_eda_visualizations(
        dataset_path=data_train_jsonl,
        output_dir=out_dir
    )

