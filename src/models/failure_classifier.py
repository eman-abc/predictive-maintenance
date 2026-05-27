"""Binary failure classification models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


class FailureClassifier:
    """Predict imminent machine failure from sensor/process features."""

    def __init__(self, model_type: str = "gbm", **kwargs):
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "gbm":
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 4),
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
        """Train classifier and return evaluation metrics."""
        self.feature_cols = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        proba = self.model.predict_proba(X_test)[:, 1]

        return {
            "f1": float(f1_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else 0.0,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_cols])

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"model": self.model, "feature_cols": self.feature_cols, "type": self.model_type},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FailureClassifier":
        data = joblib.load(path)
        instance = cls(model_type=data["type"])
        instance.model = data["model"]
        instance.feature_cols = data["feature_cols"]
        return instance
