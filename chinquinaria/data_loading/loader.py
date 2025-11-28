"""
Module: Data Loader
Loads and preprocesses the dataset from CSV.
"""

import pandas as pd

def load_data(dataset_version: str) -> pd.DataFrame:
    """
    Load the dataset from Google Drive.
    - v1_day: Data version 1 (2025-10-29), aggregation type: day
    """
    # dictionary mapping dataset versions to Google Drive file IDs
    dataset_file_ids = {
        "v1_day": "1EIqZAUtGsOI4ekDLiRoYPfLzY-vO-hfw",
        "merged_appa_eea_by_proximity_v4": "1eN3HaJX2Y9Ot_7GUW13MNh5wLfqvJewV",
        "merged_appa_eea_by_proximity_v5": "1iIOLm-jpBpZWl9kkKVxhFD1H3rrUgw1k",
        "merged_appa_eea_by_proximity_v5.5": "1tYNhoLd_bTlhWgjwyMBnQD9OGjSedFHv",
        "pm10_era5_land_era5_reanalysis_blh_final": "-"
    }

    file_id = dataset_file_ids.get(dataset_version)
    if not file_id:
        raise ValueError(f"Unknown dataset version: {dataset_version}")

    # Load the dataset from Google Drive
    if dataset_version == "pm10_era5_land_era5_reanalysis_blh_final":
        df = pd.read_csv("data/pm10_era5_land_era5_reanalysis_blh_final.csv")
    else:
        df = pd.read_csv(f"https://drive.google.com/uc?id={file_id}")

    return df