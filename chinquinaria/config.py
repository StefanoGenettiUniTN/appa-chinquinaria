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
    "dataset": "v1_day",
    "start_training_date": "2013-01-01",
    "end_training_date": "2024-01-01",
    "start_testing_date": "2024-01-01",
    "end_testing_date": "2025-01-01",
    "window_size_months": 1,
    "model_type": "xgboost",
    "llm_type": "fake",  # "open_source" or "proprietary" or "fake"
    "llm_prompt_variant_shap": "v2",  # "v1" | "v2" | "v3"
    "llm_prompt_variant_final": "v2",  # "v1" | "v2" | "v3"
    "endpoint": "https://models.inference.ai.azure.com",
    "token": os.environ.get('GITHUB_TOKEN'),
    "recycle_window_essays": False,
    "output_path": BASE_DIR / "output"
}
