"""
Evaluation metrics.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False)
    }