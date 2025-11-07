"""
Evaluation metrics.
"""
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tslearn.metrics import dtw as dtw_distance
from chinquinaria.config import CONFIG

def evaluate_predictions(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "dtw": dtw_distance(y_true, y_pred)
    }

def plot_evaluation(stazioni:pd.Series, x_date: pd.Series, y_true:pd.Series, y_pred:pd.Series):
    """Plot actual vs predicted values for each station and each month, saving each plot to disk."""
    results: pd.DataFrame = pd.DataFrame({
        "stazione": stazioni,
        "data": pd.to_datetime(x_date),
        "actual": y_true,
        "predicted": y_pred
    })
    results["year_month"] = results["data"].dt.to_period("M")
    unique_stazioni: list[str] = results["stazione"].unique().tolist()

    for staz in unique_stazioni:
        df_staz: pd.DataFrame = results[results["stazione"] == staz]

        for ym, df_month in df_staz.groupby("year_month"):
            plt.figure(figsize=(10, 4))
            plt.plot(df_month["data"].values, df_month["actual"].values, label="Actual", marker="o")
            plt.plot(df_month["data"].values, df_month["predicted"].values, label="Predicted", marker="x")
            plt.title(f"Actual vs Predicted PM10 - {staz} ({ym})")
            plt.xlabel("Date")
            plt.ylabel("Valore (PM10)")
            plt.legend()
            plt.tight_layout()

            safe_name = str(staz).replace("/", "_").replace("\\", "_").replace(" ", "_")
            file_name = f"stazione_{safe_name}_{ym}.png"
            file_path = CONFIG["output_path"] / file_name
            plt.savefig(file_path)
            plt.close()

        print(f"Saved plot for '{staz}' in {file_path}")

def plot_feature_importance(top_features: Dict[str, float], window_index: int):
    """Plot feature importance as a horizontal bar chart and save to disk."""
    plt.figure(figsize=(10, 6))
    plt.barh(list(top_features.keys()), list(top_features.values()), color='skyblue')
    plt.xlabel("Mean |SHAP| Value", fontsize=12)
    plt.title(f"Feature Importance - Window {window_index}")
    plt.tight_layout()

    file_name = f"feature_importance_window_{window_index}.png"
    file_path = CONFIG["output_path"] / file_name
    plt.savefig(file_path)
    plt.close()

    print(f"Saved feature importance plot for window {window_index} in {file_path}")