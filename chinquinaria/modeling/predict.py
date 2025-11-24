"""
Perform model inference on given test windows.
"""
import pandas as pd
import time
from chinquinaria.config import CONFIG
from chinquinaria.utils.logger import get_logger
from chinquinaria.utils.evaluation import evaluate_predictions, plot_evaluation
from lightning.pytorch.callbacks import EarlyStopping
from chinquinaria.modeling.rnn import (
    create_ts_dataset_with_covariates_for_V5_5,
    LSTModel, CONFIG_RNN, MAE
) 

logger = get_logger(__name__)

def predict(model, df: pd.DataFrame, type):
    if CONFIG["pyTorch_forecasting"]:
        if CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5.5":
            df_processed = df.copy()
            df_processed["data"] = pd.to_datetime(df_processed["Data"], format="%Y-%m-%d")
            df_processed.sort_values("Data", inplace=True)
            df_processed["time_idx"] = df_processed.groupby("Stazione_APPA").cumcount()
            df_processed.columns = df_processed.columns.str.replace('.', '_', regex=False)
            dataset = create_ts_dataset_with_covariates_for_V5_5(
                    df_processed,
                    max_encoder_length=CONFIG_RNN["max_encoder_length"],
                    max_prediction_length=CONFIG_RNN["max_prediction_length"],
                    val=False,
                    test=True,
                    preprocess=False,
            )
            print("\n\nSwitching data to dataloaders for training...\n\n")
            batch_size = 128
            df_dataloader = dataset.to_dataloader(
                train=False, batch_size=batch_size, num_workers=CONFIG_RNN["num_of_workers"]
            )
            logger.info("Dataset preprocessing DONE")
            # Execute the model prediction =============================================
            start_time = time.time()
            raw_predictions = model.predict(
                df_dataloader,
                mode="raw",
                return_x=True,
                trainer_kwargs=dict(accelerator=CONFIG_RNN["accelerator"]),
            )
            end_time = time.time()
            full_predictions_df = LSTModel.build_full_length_prediction_frame(
                raw_predictions,
                dataset,
                df_processed,
            )
            full_predictions_df["model_type"] = CONFIG["model_type"]
            predictions_file_name = f"{type}_{df_processed['data'].min().strftime('%Y%m%d')}_to_{df_processed['data'].max().strftime('%Y%m%d')}.csv"
            predictions_file_path = CONFIG["output_path"] / predictions_file_name
            full_predictions_df.to_csv(predictions_file_path, index=False)
            # Evaluate predictions =====================================================
            df_performance = evaluate_predictions(full_predictions_df["PM10_(ug_m-3)"], full_predictions_df["prediction"])
            performance_df = pd.DataFrame([{
                "model_type": CONFIG["model_type"],
                "start_date": df_processed["data"].min(),
                "end_date": df_processed["data"].max(),
                "mae": df_performance["mae"],
                "rmse": df_performance["rmse"],
                "dtw": df_performance["dtw"],
                "execution_time_seconds": round(end_time - start_time, 3)
            }])
            performance_file_path = CONFIG["output_path"] / f"{type}_{df_processed['data'].min().strftime('%Y%m%d')}_to_{df_processed['data'].max().strftime('%Y%m%d')}.csv"
            performance_df.to_csv(performance_file_path, index=False)

            return full_predictions_df

    else:
        # normalize column names
        df.columns = [col.strip().lower() for col in df.columns]
        df['data'] = pd.to_datetime(df['data'])

        # drop columns that are not features for predicting PM10
        if CONFIG["dataset"] == "v1_day":
            x_test = df.drop(columns=[  "stazione",
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
            y_test = df["valore"]
        elif CONFIG["dataset"] == "merged_appa_eea_by_proximity_v4" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5.5":
            x_test = df.drop(columns=[  "data",
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
            y_test = df["pm10_(ug.m-3)"]

            # temporary fix for missing values (only for v1_day dataset)
            if CONFIG["dataset"] == "v1_day":
                x_test = x_test.fillna(0)
                y_test = y_test.fillna(y_test.mean())

            if CONFIG["debug"]:
                logger.debug(f"\n+x_test:\n{x_test.head()}")
                logger.debug(f"\n+y_test:\n{y_test.head()}")

            # Execute the model prediction =============================================
            start_time = time.time()
            preds = model.predict(x_test)
            end_time = time.time()

            # Evaluate predictions =====================================================
            testing_performance = evaluate_predictions(y_test, preds)
            logger.info(f"Validation performance: {testing_performance}")

            # save performance to csv using pandas with columns: model_type, mae, rmse, dtw, execution_time
            performance_df = pd.DataFrame([{
                "model_type": CONFIG["model_type"],
                "start_date": df["data"].min(),
                "end_date": df["data"].max(),
                "mae": testing_performance["mae"],
                "rmse": testing_performance["rmse"],
                "dtw": testing_performance["dtw"],
                "execution_time_seconds": round(end_time - start_time, 3)
            }])
            performance_file_path = CONFIG["output_path"] / f"validation_{df['data'].min().strftime('%Y%m%d')}_to_{df['data'].max().strftime('%Y%m%d')}.csv"
            performance_df.to_csv(performance_file_path, index=False)

        # Plot evaluation ==========================================================
        if CONFIG["dataset"] == "v1_day":
            plot_evaluation(df["stazione"], df["data"], y_test, preds)
        elif CONFIG["dataset"] == "merged_appa_eea_by_proximity_v4" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5.5":
            plot_evaluation(df["stazione_appa"], df["data"], y_test, preds)

            # save predictions to csv
            if CONFIG["dataset"] == "v1_day":
                predictions_df = pd.DataFrame({
                    "model_type": CONFIG["model_type"],
                    "stazione": df["stazione"],
                    "data": df["data"],
                    "actual": y_test,
                    "predicted": preds
            })
        elif CONFIG["dataset"] == "merged_appa_eea_by_proximity_v4" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5.5":
            predictions_df = pd.DataFrame({
                "model_type": CONFIG["model_type"],
                "stazione": df["stazione_appa"],
                "data": df["data"],
                "actual": y_test,
                "predicted": preds
            })
            predictions_file_name = f"validation_{df['data'].min().strftime('%Y%m%d')}_to_{df['data'].max().strftime('%Y%m%d')}.csv"
            predictions_file_path = CONFIG["output_path"] / predictions_file_name
            predictions_df.to_csv(predictions_file_path, index=False)

            return df


def predict_windows(model, window_df: pd.DataFrame):
    # normalize column names
    window_df.columns = [col.strip().lower() for col in window_df.columns]

    # drop columns that are not features for predicting PM10
    if CONFIG["dataset"] == "v1_day":
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
    elif CONFIG["dataset"] == "merged_appa_eea_by_proximity_v4" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5.5":
        x_test = window_df.drop(columns=["data",
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
        y_test = window_df["pm10_(ug.m-3)"]

    # temporary fix for missing values (only for v1_day dataset)
    if CONFIG["dataset"] == "v1_day":
        x_test = x_test.fillna(0)
        y_test = y_test.fillna(y_test.mean())

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
    if CONFIG["dataset"] == "v1_day":
        plot_evaluation(window_df["stazione"], window_df["data"], y_test, preds)
    elif CONFIG["dataset"] == "merged_appa_eea_by_proximity_v4" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5.5":
        plot_evaluation(window_df["stazione_appa"], window_df["data"], y_test, preds)

    # save predictions to csv
    if CONFIG["dataset"] == "v1_day":
        predictions_df = pd.DataFrame({
            "model_type": CONFIG["model_type"],
            "stazione": window_df["stazione"],
            "data": window_df["data"],
            "actual": y_test,
            "predicted": preds
        })
    elif CONFIG["dataset"] == "merged_appa_eea_by_proximity_v4" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5" or CONFIG["dataset"] == "merged_appa_eea_by_proximity_v5.5":
        predictions_df = pd.DataFrame({
            "model_type": CONFIG["model_type"],
            "stazione": window_df["stazione_appa"],
            "data": window_df["data"],
            "actual": y_test,
            "predicted": preds
        })
    predictions_file_name = f"predictions_window_{window_df['data'].min().strftime('%Y%m%d')}_to_{window_df['data'].max().strftime('%Y%m%d')}.csv"
    predictions_file_path = CONFIG["output_path"] / predictions_file_name
    predictions_df.to_csv(predictions_file_path, index=False)

    return window_df
