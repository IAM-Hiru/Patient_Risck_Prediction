"""
stage04_slm_pipeline.py
-----------------------
Main runner for Stage 04 SLM Fine-Tuning Pipeline.
Orchestrates ChatML dataset preparation, 4-bit QLoRA configuration,
adapter training, and audit metric reporting.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from stage04_train_slm import train_slm_qlora

if __name__ == "__main__":
    _STAGE4_DIR = os.path.dirname(_HERE)
    dataset_path = os.path.join(_STAGE4_DIR, "data_engineer", "data", "processed", "stage04_slm_train.jsonl")
    output_model_dir = os.path.join(_HERE, "models", "stage04_slm_qlora")
    report_path = os.path.join(_HERE, "outputs", "stage04_training_report.json")

    train_slm_qlora(
        dataset_path=dataset_path,
        output_model_dir=output_model_dir,
        report_path=report_path,
        base_model="Qwen/Qwen2.5-3B-Instruct"
    )
