"""
Train the selected model using the training dataset.
"""
import pandas as pd
import time
from chinquinaria.modeling.xgboost_model import XGBoostModel
from chinquinaria.modeling.lightgbm import LightGBMModel
from chinquinaria.modeling.mlp import TorchMLPModel
from chinquinaria.utils.file_io import save_pickle
from chinquinaria.config import CONFIG
from chinquinaria.utils.evaluation import evaluate_predictions
from chinquinaria.utils.evaluation import plot_evaluation
from chinquinaria.utils.logger import get_logger

logger = get_logger(__name__)

def train_model(train_df: pd.DataFrame):
    model = None

    # Training preprocessing ===============================================
    logger.info("Training preprocessing... ")

    # normalize column names
    train_df.columns = [col.strip().lower() for col in train_df.columns]

    # drop columns that are not features for predicting PM10
    if CONFIG["dataset"] == "v1_day":
        x_train = train_df.drop(columns=[   "stazione",
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
    elif CONFIG["dataset"] == "merged_appa_eea_by_proximity_v4":
        x_train = train_df.drop(columns=["data",
                                         "stazione_appa",
                                         "pm10_(ug.m-3)",
                                         "stazione_meteo_vicina",
                                         "id_stazione_meteo_vicina",
                                         "latitudine",
                                         "longitudine",
                                         "bg_calusco_d_adda_latitudine",
                                         "bg_calusco_d_adda_longitudine",
                                         "bg_calusco_d_adda_id",
                                         "bg_osio_sotto_latitudine",
                                         "bg_osio_sotto_longitudine",
                                         "bg_osio_sotto_id",
                                         "bg_treviglio_latitudine",
                                         "bg_treviglio_longitudine",
                                         "bg_treviglio_id",
                                         "bg_via_garibaldi_latitudine",
                                         "bg_via_garibaldi_longitudine",
                                         "bg_via_garibaldi_id",
                                         "bl_area_feltrina_latitudine",
                                         "bl_area_feltrina_longitudine",
                                         "bl_area_feltrina_id",
                                         "bl_parco_città_di_bologna_latitudine",
                                         "bl_parco_città_di_bologna_longitudine",
                                         "bl_parco_città_di_bologna_id",
                                         "bl_pieve_d_alpago_latitudine",
                                         "bl_pieve_d_alpago_longitudine",
                                         "bl_pieve_d_alpago_id",
                                         "bs_palazzo_del_broletto_latitudine",
                                         "bs_palazzo_del_broletto_longitudine",
                                         "bs_palazzo_del_broletto_id",
                                         "bs_sarezzo_latitudine",
                                         "bs_sarezzo_longitudine",
                                         "bs_sarezzo_id",
                                         "cr_piazza_cadorna_latitudine",
                                         "cr_piazza_cadorna_longitudine",
                                         "cr_piazza_cadorna_id",
                                         "fe_corso_isonzo_latitudine",
                                         "fe_corso_isonzo_longitudine",
                                         "fe_corso_isonzo_id",
                                         "lc_valmadrera_latitudine",
                                         "lc_valmadrera_longitudine",
                                         "lc_valmadrera_id",
                                         "mn_ponti_sul_mincio_latitudine",
                                         "mn_ponti_sul_mincio_longitudine",
                                         "mn_ponti_sul_mincio_id",
                                         "mn_sant_agnese_latitudine",
                                         "mn_sant_agnese_longitudine",
                                         "mn_sant_agnese_id",
                                         "vr_san_bonifacio_latitudine",
                                         "vr_san_bonifacio_longitudine",
                                         "vr_san_bonifacio_id",
                                         "mo_via_ramesina_latitudine",
                                         "mo_via_ramesina_longitudine",
                                         "mo_via_ramesina_id",
                                         "pd_alta_padovana_latitudine",
                                         "pd_alta_padovana_longitudine",
                                         "pd_alta_padovana_id",
                                         "pd_arcella_latitudine",
                                         "pd_arcella_longitudine",
                                         "pd_arcella_id",
                                         "pd_este_latitudine",
                                         "pd_este_longitudine",
                                         "pd_este_id",
                                         "pd_granze_latitudine",
                                         "pd_granze_longitudine",
                                         "pd_granze_id",
                                         "pr_via_saragat_latitudine",
                                         "pr_via_saragat_longitudine",
                                         "pr_via_saragat_id",
                                         "re_san_rocco_latitudine",
                                         "re_san_rocco_longitudine",
                                         "re_san_rocco_id",
                                         "ro_largo_martiri_latitudine",
                                         "ro_largo_martiri_longitudine",
                                         "ro_largo_martiri_id",
                                         "tv_conegliano_latitudine",
                                         "tv_conegliano_longitudine",
                                         "tv_conegliano_id",
                                         "tv_mansuè_latitudine",
                                         "tv_mansuè_longitudine",
                                         "tv_mansuè_id",
                                         "tv_via_lancieri_di_novara_latitudine",
                                         "tv_via_lancieri_di_novara_longitudine",
                                         "tv_via_lancieri_di_novara_id",
                                         "ve_sacca_fisola_latitudine",
                                         "ve_sacca_fisola_longitudine",
                                         "ve_sacca_fisola_id",
                                         "ve_via_tagliamento_latitudine",
                                         "ve_via_tagliamento_longitudine",
                                         "ve_via_tagliamento_id",
                                         "vi_quartiere_italia_latitudine",
                                         "vi_quartiere_italia_longitudine",
                                         "vi_quartiere_italia_id",
                                         "vr_borgo_milano_latitudine",
                                         "vr_borgo_milano_longitudine",
                                         "vr_borgo_milano_id",
                                         "vr_bosco_chiesanuova_latitudine",
                                         "vr_bosco_chiesanuova_longitudine",
                                         "vr_bosco_chiesanuova_id",
                                         "vr_legnago_latitudine",
                                         "vr_legnago_longitudine",
                                         "vr_legnago_id"])
        y_train = train_df["pm10_(ug.m-3)"]

    # fix for missing values (only for v1_day dataset)
    if CONFIG["dataset"] == "v1_day":
        x_train = x_train.fillna(0)
        y_train = y_train.fillna(y_train.mean())

    if CONFIG["debug"]:
        logger.debug(f"\n+x_train:\n{x_train.head()}")
        logger.debug(f"\n+y_train:\n{y_train.head()}")

    logger.info("Training preprocessing DONE")

    # Model training =======================================================
    if CONFIG["model_type"] == "xgboost":
        logger.info(f"Training {CONFIG['model_type']} model...")
        model = XGBoostModel(n_estimators=200, learning_rate=0.05)
        start_time = time.time()
        model.train(x_train, y_train)
        end_time = time.time()
        save_pickle(model, f"{CONFIG['output_path']}/trained_model.pkl")
        logger.info(f"Model training DONE ({end_time - start_time:.2f} seconds) and model saved to disk ({CONFIG['output_path']}/trained_model.pkl)")
    elif CONFIG["model_type"] == "lightgbm":
        logger.info(f"Training {CONFIG['model_type']} model...")
        model = LightGBMModel(n_estimators=200, learning_rate=0.05)
        start_time = time.time()
        model.train(x_train, y_train)
        end_time = time.time()
        save_pickle(model, f"{CONFIG['output_path']}/trained_model.pkl")
        logger.info(f"Model training DONE ({end_time - start_time:.2f} seconds) and model saved to disk ({CONFIG['output_path']}/trained_model.pkl)")
    elif CONFIG["model_type"] == "mlp":
        logger.info(f"Training {CONFIG['model_type']} model...")
        model = TorchMLPModel(input_dim=x_train.shape[1])
        start_time = time.time()
        model.train(x_train, y_train)
        end_time = time.time()
        save_pickle(model, f"{CONFIG['output_path']}/trained_model.pkl")
        logger.info(f"Model training DONE ({end_time - start_time:.2f} seconds) and model saved to disk ({CONFIG['output_path']}/trained_model.pkl)")
    else:
        raise ValueError(f"Unsupported model type: {CONFIG['model_type']}")

    # Evaluate training performance ========================================
    logger.info("Evaluating training performance...")
    y_pred = model.predict(x_train)
    training_performance = evaluate_predictions(y_train, y_pred)
    logger.info("Training performance: %s", training_performance)

    # save training performance to csv using pandas with columns: model_type, mae, rmse, dtw, execution_time
    performance_df = pd.DataFrame([{
        "model_type": CONFIG["model_type"],
        "mae": training_performance["mae"],
        "rmse": training_performance["rmse"],
        "dtw": training_performance["dtw"],
        "execution_time_seconds": round(end_time - start_time, 3)
    }])
    performance_file_path = CONFIG["output_path"] / "training_performance.csv"
    performance_df.to_csv(performance_file_path, index=False)

    # Plot training predictions
    if CONFIG["dataset"] == "v1_day":
        plot_evaluation(train_df["stazione"], train_df["data"], y_train, y_pred)
        #print("Skipping plot_evaluation for v1_day dataset")
    elif CONFIG["dataset"] == "merged_appa_eea_by_proximity_v4":
        plot_evaluation(train_df["stazione_appa"], train_df["data"], y_train, y_pred)
        #print("Skipping plot_evaluation for merged_appa_eea_by_proximity_v4 dataset")

    # save training predictions to csv
    if CONFIG["dataset"] == "v1_day":
        training_predictions_df = pd.DataFrame({
            "model_type": CONFIG["model_type"],
            "stazione": train_df["stazione"],
            "data": train_df["data"],
            "actual": y_train,
            "predicted": y_pred
        })
    elif CONFIG["dataset"] == "merged_appa_eea_by_proximity_v4":
        training_predictions_df = pd.DataFrame({
            "model_type": CONFIG["model_type"],
            "stazione": train_df["stazione_appa"],
            "data": train_df["data"],
            "actual": y_train,
            "predicted": y_pred
        })
    training_predictions_file_path = CONFIG["output_path"] / "training_predictions.csv"
    training_predictions_df.to_csv(training_predictions_file_path, index=False)

    return model
