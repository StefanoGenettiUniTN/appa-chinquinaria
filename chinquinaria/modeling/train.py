"""
Train the selected model using the training dataset.
"""
import pandas as pd
from chinquinaria.modeling.xgboost_model import XGBoostModel
from chinquinaria.utils.file_io import save_pickle
from chinquinaria.config import CONFIG
from chinquinaria.utils.evaluation import evaluate_predictions
from chinquinaria.utils.evaluation import plot_evaluation
from chinquinaria.utils.logger import get_logger

logger = get_logger(__name__)

def train_model(train_df: pd.DataFrame):
    model = None

    # normalize column names
    train_df.columns = [col.strip().lower() for col in train_df.columns]

    if CONFIG["model_type"] == "xgboost":
        # Training preprocessing ===============================================
        logger.info("Training preprocessing... ")

        # drop columns that are not features for predicting PM10
        x_train = train_df.drop(columns=["stazione",
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
        y_train = train_df["valore"]
        
        if CONFIG["debug"]:
            logger.debug(f"\n+x_train:\n{x_train.head()}")
            logger.debug(f"\n+y_train:\n{y_train.head()}")

        logger.info("Training preprocessing DONE")

        # Model training =======================================================
        logger.info(f"Training {CONFIG['model_type']} model...")
        model = XGBoostModel(n_estimators=200, learning_rate=0.05)
        model.train(x_train, y_train)
        save_pickle(model, f"{CONFIG['output_path']}/trained_model.pkl")
        logger.info(f"Model training DONE and model saved to disk ({CONFIG['output_path']}/trained_model.pkl)")
        
        # Evaluate training performance ========================================
        logger.info("Evaluating training performance...")
        y_pred = model.predict(x_train)
        training_performance = evaluate_predictions(y_train, y_pred)
        logger.info("Training performance: %s", training_performance)

        # Plot training predictions
        plot_evaluation(train_df["stazione"], train_df["data"], y_train, y_pred)
    else:
        raise ValueError(f"Unsupported model type: {CONFIG['model_type']}")

    return model
