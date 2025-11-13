import pandas as pd
import optuna
import optuna.visualization as vis
import torch
import numpy as np
from typing import Dict, List, Set, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from chinquinaria.data_loading.loader import load_data
from chinquinaria.config import CONFIG
from chinquinaria.utils.logger import get_logger
from chinquinaria.data_loading.splitter import split_train_test
from chinquinaria.modeling.xgboost_model import XGBoostModel
from chinquinaria.modeling.lightgbm import LightGBMModel
from chinquinaria.modeling.mlp import MLPModel
from chinquinaria.modeling.mlp import TorchMLPModel
from chinquinaria.modeling.random_forest import RandomForestModel

logger = get_logger(__name__)

def objective(trial: optuna.trial.Trial, x: pd.DataFrame, y: pd.Series):
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if CONFIG["model_type"] == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        model = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth)
        model.train(x_train, y_train)
        preds = model.predict(x_valid)
    elif CONFIG["model_type"] == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10)
        }
        model = XGBoostModel(**params)
        model.train(x_train, y_train)
        preds = model.predict(x_valid)
    elif CONFIG["model_type"] == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150)
        }
        model = LightGBMModel(**params)
        model.train(x_train, y_train)
        preds = model.predict(x_valid)
    elif CONFIG["model_type"] == "mlp":
        hidden1 = trial.suggest_int("hidden1", 32, 256, log=True)
        hidden2 = trial.suggest_int("hidden2", 16, 128, log=True)
        hidden3 = trial.suggest_int("hidden3", 8, 64, log=True)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 20, 80)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = TorchMLPModel(input_dim=x.shape[1])
        model.optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
        model.batch_size = batch_size
        model.epochs = epochs
        model.model = MLPModel(
            input_dim=x.shape[1],
            hidden_layers=[hidden1, hidden2, hidden3]
        ).to(model.device)
        model.train(x_train, y_train)
        preds = model.predict(x_valid)
    else:
        raise ValueError(f"Unsupported model type: {CONFIG['model_type']}")

    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse
    
def run_optuna(x: pd.DataFrame, y: pd.Series, n_trials=30):
    logger.info(f"Starting Optuna hyperparameter optimization with {n_trials} trials for {CONFIG['model_type']}...")
    study = optuna.create_study(direction="minimize", study_name=f"{CONFIG['model_type']}_optuna_study")
    study.optimize(lambda trial: objective(trial, x, y), n_trials=n_trials)

    logger.info("Optuna optimization completed.")
    logger.info(f"Best trial parameters: {study.best_trial.params}")
    logger.info(f"Best trial value (RMSE): {study.best_trial.value:.4f}")
    study.trials_dataframe().to_csv(f"{CONFIG['output_path']}/{CONFIG['model_type']}_optuna_results.csv", index=False)
    fig = vis.plot_optimization_history(study)
    fig.write_html(f"{CONFIG['output_path']}/{CONFIG['model_type']}_optuna_optimization_history.html")
    fig = vis.plot_param_importances(study)
    fig.write_html(f"{CONFIG['output_path']}/{CONFIG['model_type']}_optuna_param_importance.html")
    fig = vis.plot_slice(study)
    fig.write_html(f"{CONFIG['output_path']}/{CONFIG['model_type']}_optuna_slice.html")
    fig = vis.plot_parallel_coordinate(study)
    fig.write_html(f"{CONFIG['output_path']}/{CONFIG['model_type']}_optuna_parallel_coordinate.html")
    return study

if __name__ == "__main__":
    # load dataset =============================================================
    df = load_data(CONFIG["dataset"])

    if df.empty:
        logger.error("Data loading failed or returned empty dataset.")
    else:
        logger.info(f"Data loaded successfully with {len(df)} records.")
        if CONFIG["debug"]:
            for i, col in enumerate(df.columns):
                logger.debug(f"Column [{i}]: {col}")
            logger.debug(f"Data sample:\n{df.head()}")

    # simple preprocessing =====================================================
    # normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # drop columns that are not features for predicting PM10
    x_optuna = df.drop(columns=[  "stazione",
                            "data",
                            "inquinante",
                            "unità di misura",
                            "latitudine",
                            "longitudine",
                            "valore",
                            "nazione",
                            "comune",
                            "stazionemeteo",
                            "lombardia_ma_0_min_lat",
                            "lombardia_ma_0_min_lon",
                            "lombardia_ma_0_max_lat",
                            "lombardia_ma_0_max_lon",
                            "lombardia_ma_0_unità di misura",
                            "lombardia_ma_1_min_lon",
                            "lombardia_ma_1_min_lat",
                            "lombardia_ma_1_max_lat",
                            "lombardia_ma_1_max_lon",
                            "lombardia_ma_1_unità di misura",
                            "lombardia_ma_2_min_lon",
                            "lombardia_ma_2_min_lat",
                            "lombardia_ma_2_max_lat",
                            "lombardia_ma_2_max_lon",
                            "lombardia_ma_2_unità di misura",
                            "lombardia_ma_3_min_lon",
                            "lombardia_ma_3_min_lat",
                            "lombardia_ma_3_max_lat",
                            "lombardia_ma_3_max_lon",
                            "lombardia_ma_3_unità di misura",
                            "lombardia_ma_4_min_lon",
                            "lombardia_ma_4_min_lat",
                            "lombardia_ma_4_max_lat",
                            "lombardia_ma_4_max_lon",
                            "lombardia_ma_4_unità di misura",
                            "lombardia_ma_5_min_lon",
                            "lombardia_ma_5_min_lat",
                            "lombardia_ma_5_max_lat",
                            "lombardia_ma_5_max_lon",
                            "lombardia_ma_5_unità di misura",
                            "lombardia_ma_6_min_lon",
                            "lombardia_ma_6_min_lat",
                            "lombardia_ma_6_max_lat",
                            "lombardia_ma_6_max_lon",
                            "lombardia_ma_6_unità di misura",
                            "lombardia_ma_7_min_lon",
                            "lombardia_ma_7_min_lat",
                            "lombardia_ma_7_max_lat",
                            "lombardia_ma_7_max_lon",
                            "lombardia_ma_7_unità di misura",
                            "lombardia_ma_8_min_lon",
                            "lombardia_ma_8_min_lat",
                            "lombardia_ma_8_max_lat",
                            "lombardia_ma_8_max_lon",
                            "lombardia_ma_8_unità di misura",
                            "lombardia_ma_9_min_lon",
                            "lombardia_ma_9_min_lat",
                            "lombardia_ma_9_max_lat",
                            "lombardia_ma_9_max_lon",
                            "lombardia_ma_9_unità di misura",
                            "lombardia_ma_10_min_lon",
                            "lombardia_ma_10_min_lat",
                            "lombardia_ma_10_max_lat",
                            "lombardia_ma_10_max_lon",
                            "lombardia_ma_10_unità di misura",
                            "lombardia_ma_11_min_lon",
                            "lombardia_ma_11_min_lat",
                            "lombardia_ma_11_max_lat",
                            "lombardia_ma_11_max_lon",
                            "lombardia_ma_11_unità di misura",
                            "lombardia_ma_12_min_lon",
                            "lombardia_ma_12_min_lat",
                            "lombardia_ma_12_max_lat",
                            "lombardia_ma_12_max_lon",
                            "lombardia_ma_12_unità di misura",
                            "lombardia_ma_13_min_lon",
                            "lombardia_ma_13_min_lat",
                            "lombardia_ma_13_max_lat",
                            "lombardia_ma_13_max_lon",
                            "lombardia_ma_13_unità di misura",
                            "lombardia_ma_14_min_lon",
                            "lombardia_ma_14_min_lat",
                            "lombardia_ma_14_max_lat",
                            "lombardia_ma_14_max_lon",
                            "lombardia_ma_14_unità di misura",
                            "lombardia_ma_15_min_lon",
                            "lombardia_ma_15_min_lat",
                            "lombardia_ma_15_max_lat",
                            "lombardia_ma_15_max_lon",
                            "lombardia_ma_15_unità di misura",
                            "lombardia_ma_16_min_lon",
                            "lombardia_ma_16_min_lat",
                            "lombardia_ma_16_max_lat",
                            "lombardia_ma_16_max_lon",
                            "lombardia_ma_16_unità di misura",
                            "lombardia_ma_17_min_lon",
                            "lombardia_ma_17_min_lat",
                            "lombardia_ma_17_max_lat",
                            "lombardia_ma_17_max_lon",
                            "lombardia_ma_17_unità di misura",
                            "veneto_ma_0_min_lon",
                            "veneto_ma_0_min_lat",
                            "veneto_ma_0_max_lat",
                            "veneto_ma_0_max_lon",
                            "veneto_ma_0_unità di misura",
                            "veneto_ma_1_min_lon",
                            "veneto_ma_1_min_lat",
                            "veneto_ma_1_max_lat",
                            "veneto_ma_1_max_lon",
                            "veneto_ma_1_unità di misura",
                            "veneto_ma_2_min_lon",
                            "veneto_ma_2_min_lat",
                            "veneto_ma_2_max_lat",
                            "veneto_ma_2_max_lon",
                            "veneto_ma_2_unità di misura",
                            "veneto_ma_6_min_lon",
                            "veneto_ma_6_min_lat",
                            "veneto_ma_6_max_lat",
                            "veneto_ma_6_max_lon",
                            "veneto_ma_6_unità di misura",
                            "veneto_ma_7_min_lon",
                            "veneto_ma_7_min_lat",
                            "veneto_ma_7_max_lat",
                            "veneto_ma_7_max_lon",
                            "veneto_ma_7_unità di misura",
                            "veneto_ma_8_min_lon",
                            "veneto_ma_8_min_lat",
                            "veneto_ma_8_max_lat",
                            "veneto_ma_8_max_lon",
                            "veneto_ma_8_unità di misura",
                            "veneto_ma_9_min_lon",
                            "veneto_ma_9_min_lat",
                            "veneto_ma_9_max_lat",
                            "veneto_ma_9_max_lon",
                            "veneto_ma_9_unità di misura",
                            "veneto_ma_11_min_lon",
                            "veneto_ma_11_min_lat",
                            "veneto_ma_11_max_lat",
                            "veneto_ma_11_max_lon",
                            "veneto_ma_11_unità di misura",
                            "veneto_ma_12_min_lon",
                            "veneto_ma_12_min_lat",
                            "veneto_ma_12_max_lat",
                            "veneto_ma_12_max_lon",
                            "veneto_ma_12_unità di misura",
                            "veneto_ma_13_min_lon",
                            "veneto_ma_13_min_lat",
                            "veneto_ma_13_max_lat",
                            "veneto_ma_13_max_lon",
                            "veneto_ma_13_unità di misura",
                            "veneto_ma_14_min_lon",
                            "veneto_ma_14_min_lat",
                            "veneto_ma_14_max_lat",
                            "veneto_ma_14_max_lon",
                            "veneto_ma_14_unità di misura",
                            "veneto_ma_15_min_lon",
                            "veneto_ma_15_min_lat",
                            "veneto_ma_15_max_lat",
                            "veneto_ma_15_max_lon",
                            "veneto_ma_15_unità di misura",
                            "veneto_ma_16_min_lon",
                            "veneto_ma_16_min_lat",
                            "veneto_ma_16_max_lat",
                            "veneto_ma_16_max_lon",
                            "veneto_ma_16_unità di misura",
                            "veneto_ma_17_min_lon",
                            "veneto_ma_17_min_lat",
                            "veneto_ma_17_max_lat",
                            "veneto_ma_17_max_lon",
                            "veneto_ma_17_unità di misura",
                            "veneto_ma_18_min_lon",
                            "veneto_ma_18_min_lat",
                            "veneto_ma_18_max_lat",
                            "veneto_ma_18_max_lon",
                            "veneto_ma_18_unità di misura"])
    y_optuna = df["valore"]
    
    # temporary fix for missing values
    x_optuna = x_optuna.fillna(0)
    y_optuna = y_optuna.fillna(y_optuna.mean())

    # hyperparameter optimization ==============================================
    study = run_optuna(x_optuna, y_optuna, n_trials=50)