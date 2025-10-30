"""
Perform model inference on given test windows.
"""

def predict_windows(model, window_df):
    X_test = window_df.drop(columns=["PM10"])
    preds = model.predict(X_test)
    window_df = window_df.copy()
    window_df["predicted_PM10"] = preds
    return window_df
