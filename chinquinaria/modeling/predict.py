"""
Perform model inference on given test windows.
"""
import pandas as pd
import time
from chinquinaria.config import CONFIG
from chinquinaria.utils.logger import get_logger
from chinquinaria.utils.evaluation import evaluate_predictions, plot_evaluation

logger = get_logger(__name__)

def predict_windows(model, window_df: pd.DataFrame):
    # normalize column names
    window_df.columns = [col.strip().lower() for col in window_df.columns]

    # drop columns that are not features for predicting PM10
    x_test = window_df.drop(columns=[   "stazione",
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
    y_test = window_df["valore"]

    if CONFIG["debug"]:
        logger.debug(f"\n+x_test:\n{x_test.head()}")
        logger.debug(f"\n+y_test:\n{y_test.head()}")

    # Execute the model prediction =============================================
    start_time = time.time()
    preds = model.predict(x_test)
    end_time = time.time()

    # Evaluate predictions on the window =======================================
    testing_window_performance = evaluate_predictions(y_test, preds)
    logger.info(f"Testing window performance: {testing_window_performance}")

    # save window performance to csv using pandas with columns: model_type, mae, rmse, dtw, execution_time
    performance_df = pd.DataFrame([{
        "model_type": CONFIG["model_type"],
        "window_start_date": window_df["data"].min(),
        "window_end_date": window_df["data"].max(),
        "mae": testing_window_performance["mae"],
        "rmse": testing_window_performance["rmse"],
        "dtw": testing_window_performance["dtw"],
        "execution_time_seconds": round(end_time - start_time, 3)
    }])
    performance_file_path = CONFIG["output_path"] / f"performance_window_{window_df['data'].min().strftime('%Y%m%d')}_to_{window_df['data'].max().strftime('%Y%m%d')}.csv"
    performance_df.to_csv(performance_file_path, index=False)

    # Plot evaluation for the window ===========================================
    plot_evaluation(window_df["stazione"], window_df["data"], y_test, preds)

    # save predictions to csv
    predictions_df = pd.DataFrame({
        "model_type": CONFIG["model_type"],
        "stazione": window_df["stazione"],
        "data": window_df["data"],
        "actual": y_test,
        "predicted": preds
    })
    predictions_file_name = f"predictions_window_{window_df['data'].min().strftime('%Y%m%d')}_to_{window_df['data'].max().strftime('%Y%m%d')}.csv"
    predictions_file_path = CONFIG["output_path"] / predictions_file_name
    predictions_df.to_csv(predictions_file_path, index=False)

    return window_df
