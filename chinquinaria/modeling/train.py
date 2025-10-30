"""
Train the selected model using the training dataset.
"""

from xgboost_model import XGBoostModel
from chinquinaria.utils.file_io import save_pickle

def train_model(train_df):
    X_train = train_df.drop(columns=["PM10"])
    y_train = train_df["PM10"]

    model = XGBoostModel(n_estimators=200, learning_rate=0.05)
    model.train(X_train, y_train)
    save_pickle(model, "trained_model.pkl")

    return model
