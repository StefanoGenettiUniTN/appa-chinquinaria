"""
Central configuration file.
Use this to define paths, constants, and default model parameters.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent


###############################################################
# IMPORTANT NOTE: the pytorch-forecasting models (e.g. LSTM)
# require at least `max_encoder_length` days of history before
# they can start making predictions for each station.
#
# This means the first `max_encoder_length` days of your test set
# will NOT be predicted. For example, with the default encoder length
# of 30, predictions will only start after the first 30 days.
#
# To ensure you get predictions for your desired test period,
# you should start your test set at least `max_encoder_length` days
# before the first date you want predictions for.
#
# Similarly, you may want to trim the first `max_encoder_length` days
# from your validation set if you want to evaluate only the predicted period.
#
# Example: If you want predictions starting from 2024-01-01 and
# encoder length is 30, set `start_testing_date` to 2023-12-01 or earlier.
#
# See also: PyTorch Forecasting TimeSeriesDataSet documentation.
#
# Below an example of lstm dates choices considering an encoder_lenght of '30':
#    "start_training_date": "2013-01-01",
#    "end_training_date": "2023-01-01",
#    "start_validation_date": "2023-01-02",
#    "end_validation_date": "2023-12-01",
#    "start_testing_date": "2023-12-02",
#    "end_testing_date": "2025-01-01",
#
# and the current dates to use with models not of the pytorch-forecasting frameworks:
#    "start_training_date": "2013-01-01",
#    "end_training_date": "2023-01-01",
#    "start_validation_date": "2023-01-01",
#    "end_validation_date": "2024-01-01",
#    "start_testing_date": "2024-01-01",
#    "end_testing_date": "2025-01-01",
# This is highly hard-coded and should be changed i further improvements
###############################################################
CONFIG = {
    "debug": True,
    "dataset": "merged_appa_eea_by_proximity_v5.5", # v1_day or merged_appa_eea_by_proximity_v4 or merged_appa_eea_by_proximity_v5 or merged_appa_eea_by_proximity_v5.5 or pm10_era5_land_era5_reanalysis_blh_final
    "start_training_date": "2013-01-01",
    "end_training_date": "2023-01-01",
    "start_validation_date": "2023-01-02",
    "end_validation_date": "2023-12-01",
    "start_testing_date": "2023-12-02",
    "end_testing_date": "2025-01-01",
    "window_size_months": 1,
    "model_type": "lstm", # "xgboost" or "lightgbm" or "mlp" or "random_forest" (not implemented) or "eldt" (not implemented) or "lstm"
    "pyTorch_forecasting":True, # True if "model_type" is in ["lstm",] otherwise False
    "llm_type": "fake",  # "open_source" or "proprietary" or "fake"
    "endpoint": "https://models.inference.ai.azure.com",
    "token": os.environ.get('GITHUB_TOKEN'),
    "llm_prompt_variant_shap": "v4",  # "v1", "v2", "v3", "v4"
    "llm_prompt_variant_final": "v4",  # "v1", "v2", "v3", "v4"
    "recycle_window_essays": False,
    "output_path": BASE_DIR / "output",
}
