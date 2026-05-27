"""RUL regression models using Random Forest and Gradient Boosting."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.utils.metrics import rul_score


class RULRegressor:
    """Remaining Useful Life regressor with RF and GBM backends."""

    def __init__(self, model_type: str = "rf", **kwargs):
        if model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 15),
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "gbm":
            self.model = GradientBoostingRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.model_type = model_type
        self.feature_cols: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> dict:
        """Train the regressor and return evaluation metrics."""
        self.feature_cols = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        return {
            "rmse": float(np.sqrt(np.mean((preds - y_test) ** 2))),
            "mae": float(np.mean(np.abs(preds - y_test))),
            "rul_score": rul_score(y_test.values, preds),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_cols])

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"model": self.model, "feature_cols": self.feature_cols, "type": self.model_type},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RULRegressor":
        data = joblib.load(path)
        instance = cls(model_type=data["type"])
        instance.model = data["model"]
        instance.feature_cols = data["feature_cols"]
        return instance
