"""
Central configuration file.
Use this to define paths, constants, and default model parameters.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "debug": True,
    "dataset": "merged_appa_eea_by_proximity_v5.5", # v1_day or merged_appa_eea_by_proximity_v4 or merged_appa_eea_by_proximity_v5 or merged_appa_eea_by_proximity_v5.5
    "start_training_date": "2013-01-01",
    "end_training_date": "2023-01-01",
    "start_validation_date": "2023-01-01",
    "end_validation_date": "2024-01-01",
    "start_testing_date": "2024-01-01",
    "end_testing_date": "2025-01-01",
    "window_size_months": 1,
    "model_type": "lstm", # "xgboost" or "lightgbm" or "mlp" or "random_forest" (not implemented) or "eldt" (not implemented) or "lstm" (not implemented)
    "llm_type": "fake",  # "open_source" or "proprietary" or "fake"
    "endpoint": "https://models.inference.ai.azure.com",
    "token": os.environ.get('GITHUB_TOKEN'),
    "llm_prompt_variant_shap": "v4",  # "v1", "v2", "v3", "v4"
    "llm_prompt_variant_final": "v4",  # "v1", "v2", "v3", "v4"
    "recycle_window_essays": False,
    "output_path": BASE_DIR / "output",
    "pyTorch_forecasting":True,
}
