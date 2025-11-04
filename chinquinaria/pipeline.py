"""
Main orchestration script for the end-to-end pollutant forecasting pipeline.
"""
import pandas as pd
from typing import Dict, List, Set, Tuple
from chinquinaria.config import CONFIG
from chinquinaria.data_loading.loader import load_data
from chinquinaria.data_loading.splitter import split_train_test, create_time_windows
from chinquinaria.modeling.train import train_model
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

    train_df, test_df = split_train_test(df, CONFIG["start_training_date"], CONFIG["end_training_date"],
                                        CONFIG["start_testing_date"], CONFIG["end_testing_date"])

    if CONFIG["debug"]:
        logger.debug(f"Training data length: {len(train_df)}")
        logger.debug(f"Training data date range: {train_df['Data'].min()} to {train_df['Data'].max()}")
        logger.debug(f"Testing data length: {len(test_df)}")
        logger.debug(f"Testing data date range: {test_df['Data'].min()} to {test_df['Data'].max()}")

    logger.info("Training model...")
    model = train_model(train_df)
    logger.info("Model training completed.")

    windows: List[pd.DataFrame] = create_time_windows(test_df, CONFIG["window_size_months"])
    window_summaries = []
    
    if CONFIG["debug"]:
        logger.debug(f"Number of time windows created: {len(windows)}")
        for i, window in enumerate(windows):
            logger.debug(f"Window {i+1} length: {len(window)}")
            logger.debug(f"Window {i+1} date range: {window['Data'].min()} to {window['Data'].max()}")

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

        shap_text = generate_shap_summary(shap_res, window["data"].min(), window["data"].max())
        summary_text = summarize_shap(shap_text)

        if CONFIG["debug"]:
            logger.debug(f"LLM Summary for window {i}:\n{summary_text}")

        window_summaries.append(summary_text)

    exit(0)

    logger.info("Generating final report...")
    final_report = generate_final_essay(window_summaries, CONFIG["llm_type"])

    with open(CONFIG["output_path"], "w") as f:
        f.write(final_report)

    logger.info(f"Pipeline completed! Report saved to {CONFIG['output_path']}")

if __name__ == "__main__":
    run_pipeline()
