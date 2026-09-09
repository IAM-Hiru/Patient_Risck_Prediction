"""
stage04_etl_pipeline.py
-----------------------
Main runner for Stage 04 Data Engineer ETL Transformation Pipeline.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from stage04_transform import run_stage04_etl

if __name__ == "__main__":
    run_stage04_etl()
