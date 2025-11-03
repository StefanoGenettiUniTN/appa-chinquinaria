"""
Central configuration file.
Use this to define paths, constants, and default model parameters.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "debug": True,
    "dataset": "v1_day",
    "start_training_date": "2013-01-01",
    "end_training_date": "2024-01-01",
    "start_testing_date": "2024-01-01",
    "end_testing_date": "2025-01-01",
    "window_size_months": 1,
    "model_type": "xgboost",
    "llm_type": "open_source",  # or "proprietary"
    "output_path": BASE_DIR / "output"
}
