"""Unsupervised anomaly detection for early degradation signals (UC5 Component B-a)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


class AnomalyDetector:
    """
    Isolation Forest trained on relatively healthy engine cycles (high RUL).

    Higher ``anomaly_score`` (0–100) = more unusual / degraded vs training baseline.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols: list[str] = []
        self.min_rul_fit: int = 30
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def _fit_matrix(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        return df[feature_cols].fillna(0).values

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        *,
        min_rul: int = 30,
        max_rows: int = 100_000,
    ) -> dict[str, float]:
        """
        Fit on training rows with RUL >= min_rul (proxy for normal operation).
        """
        self.feature_cols = feature_cols
        self.min_rul_fit = min_rul
        healthy = df[df["rul"] >= min_rul]
        if len(healthy) == 0:
            healthy = df
        if len(healthy) > max_rows:
            healthy = healthy.sample(n=max_rows, random_state=42)
        X = self._fit_matrix(healthy, feature_cols)
        self.model.fit(X)
        raw = -self.model.score_samples(X)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + 1e-6
        preds = self.model.predict(X)
        return {
            "train_rows": float(len(healthy)),
            "train_pct_flagged": float((preds == -1).mean()),
        }

    def _raw_risk(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)

    def score_to_0_100(self, raw: np.ndarray) -> np.ndarray:
        scaled = (raw - self._score_min) / (self._score_max - self._score_min)
        return np.clip(scaled * 100.0, 0.0, 100.0)

    def predict_scores(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (anomaly_score 0–100, is_anomaly 0/1)."""
        X = self._fit_matrix(df, self.feature_cols)
        raw = self._raw_risk(X)
        scores = self.score_to_0_100(raw)
        flags = (self.model.predict(X) == -1).astype(int)
        return scores, flags

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "min_rul_fit": self.min_rul_fit,
                "score_min": self._score_min,
                "score_max": self._score_max,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_cols = data["feature_cols"]
        instance.min_rul_fit = data.get("min_rul_fit", 30)
        instance._score_min = data["score_min"]
        instance._score_max = data["score_max"]
        return instance


def evaluate_anomaly_degradation_proxy(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    *,
    rul_threshold: int = 30,
    eval_protocol: str,
) -> dict[str, float | int | str]:
    """
    Proxy evaluation: high anomaly score should align with low RUL (degraded).
    """
    scores = np.asarray(anomaly_scores, dtype=float)
    y = (df["rul"].values <= rul_threshold).astype(int)
    out: dict[str, float | int | str] = {
        "eval_protocol": eval_protocol,
        "n_samples": int(len(df)),
        "mean_anomaly_score": float(np.mean(scores)),
        "pct_flagged": float(np.mean(scores >= 50.0)),
    }
    if len(np.unique(y)) > 1:
        out["degradation_roc_auc"] = float(roc_auc_score(y, scores))
    return out
