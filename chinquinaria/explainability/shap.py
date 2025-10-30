"""
run_shap: Compute SHAP values for model predictions per time window.
generate_shap_summary: Generate a textual summary or JSON report from SHAP analysis results.
"""

import shap
import pandas as pd

def run_shap(model, window_df, target_col="PM10"):
    X = window_df.drop(columns=[target_col])
    explainer = shap.Explainer(model.model)
    shap_values = explainer(X)
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    return {
        "mean_shap": mean_abs,
        "top_features": mean_abs.head(5).to_dict()
    }

def generate_shap_summary(shap_result, window_index):
    text = f"""
    SHAP Summary for Window {window_index}:
    Top Influential Features:
    {shap_result['top_features']}
    """
    return text.strip()