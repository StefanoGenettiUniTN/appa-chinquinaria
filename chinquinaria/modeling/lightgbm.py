"""
LightGBM model implementation for pollutant forecasting.
"""

import lightgbm as lgb

class LightGBMModel:
    def __init__(self, **params):
        self.model = lgb.LGBMRegressor(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def __call__(self, X):
        return self.predict(X)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)
