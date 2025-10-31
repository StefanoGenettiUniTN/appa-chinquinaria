"""
Main orchestration script for the end-to-end pollutant forecasting pipeline.
"""

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

logger = get_logger(__name__)

def run_pipeline():
    logger.info("Loading data...")
    df = load_data(CONFIG["dataset"])

    # check if data is loaded
    if df.empty:
        logger.error("Data loading failed or returned empty dataset.")
    else:
        logger.info(f"Data loaded successfully with {len(df)} records.")
        # columns
        for col in df.columns:
            logger.info(f"Column: {col}")
        # sample data
        logger.info(f"Data sample:\n{df.head()}")

    exit(0)

    train_df, test_df = split_train_test(df, CONFIG["split_date"])

    logger.info("Training model...")
    model = train_model(train_df, CONFIG["target"])

    windows = create_time_windows(test_df, CONFIG["window_size_months"])
    window_summaries = []

    for i, window in enumerate(windows, start=1):
        logger.info(f"Processing window {i}/{len(windows)}...")
        preds_df = predict_windows(model, window)
        shap_res = run_shap(model, preds_df)
        shap_text = generate_shap_summary(shap_res, i)
        summary_text = summarize_shap(shap_text, CONFIG["llm_type"])
        window_summaries.append(summary_text)

    logger.info("Generating final report...")
    final_report = generate_final_essay(window_summaries, CONFIG["llm_type"])

    with open(CONFIG["output_path"], "w") as f:
        f.write(final_report)

    logger.info(f"Pipeline completed! Report saved to {CONFIG['output_path']}")

if __name__ == "__main__":
    run_pipeline()
