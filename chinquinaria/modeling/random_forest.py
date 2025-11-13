"""
RandomForest model implementation for pollutant forecasting.
"""

from sklearn.ensemble import RandomForestRegressor
import joblib  # per salvare e caricare il modello

class RandomForestModel:
    def __init__(self, **params):
        self.model = RandomForestRegressor(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def __call__(self, X):
        return self.predict(X)

    def save(self, path):
        # Usa joblib per salvare il modello sklearn
        joblib.dump(self.model, path)

    def load(self, path):
        # Ricarica il modello
        self.model = joblib.load(path)
