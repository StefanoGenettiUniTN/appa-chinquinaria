"""
Main orchestration script for the end-to-end pollutant forecasting pipeline.
"""
import pandas as pd
from typing import Dict, List, Set, Tuple
from chinquinaria.config import CONFIG
from chinquinaria.data_loading.loader import load_data
from chinquinaria.data_loading.splitter import split_train_test_validation, create_time_windows
from chinquinaria.modeling.train import train_model
from chinquinaria.modeling.predict import predict
from chinquinaria.modeling.predict import predict_windows
from chinquinaria.explainability.shap import run_shap
from chinquinaria.explainability.shap import generate_shap_summary
from chinquinaria.llm_reporting.llm import summarize_shap
from chinquinaria.llm_reporting.llm import generate_final_essay
from chinquinaria.utils.logger import get_logger
from chinquinaria.utils.evaluation import plot_feature_importance

logger = get_logger(__name__)

def run_pipeline():
    logger.info("Loading data...")
    df = load_data(CONFIG["dataset"])

    if df.empty:
        logger.error("Data loading failed or returned empty dataset.")
    else:
        logger.info(f"Data loaded successfully with {len(df)} records.")
        if CONFIG["debug"]:
            for col in df.columns:
                logger.debug(f"Column: {col}")
            logger.debug(f"Data sample:\n{df.head()}")

    train_df, validation_df, test_df = split_train_test_validation( df=df,
                                                                    training_start_date=CONFIG["start_training_date"],
                                                                    training_end_date=CONFIG["end_training_date"],
                                                                    validation_start_date=CONFIG["start_validation_date"],
                                                                    validation_end_date=CONFIG["end_validation_date"],
                                                                    testing_start_date=CONFIG["start_testing_date"],
                                                                    testing_end_date=CONFIG["end_testing_date"])
    if CONFIG["debug"]:
        logger.debug(f"Training data length: {len(train_df)}")
        logger.debug(f"Training data date range: {train_df['Data'].min()} to {train_df['Data'].max()}")
        logger.debug(f"Testing data length: {len(test_df)}")
        logger.debug(f"Testing data date range: {test_df['Data'].min()} to {test_df['Data'].max()}")

    logger.info("Training model...")
    model = train_model(train_df)
    logger.info("Model training completed.")

    logger.info("Validation set evaluation...")
    validation_preds_df = predict(model, validation_df, type="validation")
    logger.info("Validation completed.")

    windows: List[pd.DataFrame] = create_time_windows(test_df, CONFIG["window_size_months"])
    window_summaries = []
    shap_texts = []
    
    if CONFIG["debug"]:
        logger.debug(f"Number of time windows created: {len(windows)}")
        for i, window in enumerate(windows):
            logger.debug(f"Window {i+1} length: {len(window)}")
            logger.debug(f"Window {i+1} date range: {window['Data'].min()} to {window['Data'].max()}")

    if CONFIG["pyTorch_forecasting"]:
        test_predictions_df = predict(model, test_df, type="test")
        for i, window in enumerate(windows, start=1):
            logger.info(f"Processing window {i}/{len(windows)}...")
            curr_preds_df = test_predictions_df[test_predictions_df["data"].isin(window["Data"].unique())]
            model.plot_full_length_predictions(curr_preds_df)
    
    else:
        for i, window in enumerate(windows, start=1):
            logger.info(f"Processing window {i}/{len(windows)}...")
            preds_df: pd.DataFrame = predict_windows(model, window)
            shap_res = run_shap(model, preds_df)
            
            if CONFIG["debug"]:
                logger.debug(f"SHAP results for window {i}: {shap_res}")

            plot_feature_importance(
                top_features=shap_res["top_features"],
                window_index=i
            )

            # save shap results to csv with columns: window_id, feature, mean_abs_shap_value
            mean_shap = shap_res["mean_shap"]
            shap_df = pd.DataFrame({
                "model_type": CONFIG["model_type"],
                "window_id": i,
                "feature": mean_shap.index,
                "mean_abs_shap_value": mean_shap.values
            })
            shap_file_name = f"shap_values_window_{i}.csv"
            shap_file_path = CONFIG["output_path"] / shap_file_name
            shap_df.to_csv(shap_file_path, index=False)

            if CONFIG["recycle_window_essays"]:
                summary_file_name = f"llm_summary_window_{i}.txt"
                summary_file_path = CONFIG["output_path"] / summary_file_name
                summary_file = open(summary_file_path, "r")
                summary_text = summary_file.read()
                summary_file.close()
                logger.info(f"Recycled LLM summary for window {i} from {summary_file_path}")
            else:
                shap_text = generate_shap_summary(shap_res, window["data"].min(), window["data"].max())

                # save shap_text on file
                shap_file_name = f"shap_summary_window_{i}.txt"
                shap_file_path = CONFIG["output_path"] / shap_file_name
                shap_file = open(shap_file_path, "w")
                shap_file.write(shap_text)
                shap_file.close()

            shap_texts.append(shap_text)
            summary_text = summarize_shap(shap_text)

                if CONFIG["debug"]:
                    logger.debug(f"LLM Summary for window {i}:\n{summary_text}")

                summary_file_name = f"llm_summary_window_{i}.txt"
                summary_file_path = CONFIG["output_path"] / summary_file_name
                summary_file = open(summary_file_path, "w")
                summary_file.write(summary_text)
                summary_file.close()

            window_summaries.append(summary_text)

    logger.info("Generating final report...")
    shap_corpus_text = "\n\n".join(shap_texts) if shap_texts else None
    final_report = generate_final_essay(window_summaries, shap_data=shap_corpus_text)

            final_report_file_name = "final_report.txt"
            final_report_file_path = CONFIG["output_path"] / final_report_file_name
            final_report_file = open(final_report_file_path, "w")
            final_report_file.write(final_report)
            final_report_file.close()

    logger.info(f"Pipeline completed! Report saved to {CONFIG['output_path']}")

if __name__ == "__main__":
    run_pipeline()
