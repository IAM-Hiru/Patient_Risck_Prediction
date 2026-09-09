"""
stage04_integration_pipeline.py
--------------------------------
Main runner for Stage 04 Systems Integration & QA Pipeline.
Generates test data payloads if needed, runs end-to-end integration test across 50 records,
validates schema boundaries, and outputs verification report.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from generate_stage03_data import OUT_FILE
from stage04_integration_test import run_integration_pipeline_test

if __name__ == "__main__":
    report_path = os.path.join(_HERE, "outputs", "stage04_integration_report.json")
    run_integration_pipeline_test(
        stage3_inputs_path=OUT_FILE,
        report_output_path=report_path
    )
