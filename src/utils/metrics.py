"""Custom evaluation metrics for predictive maintenance models."""

import numpy as np


def rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM scoring function for RUL predictions.
    Asymmetric penalty: over-estimation penalized less than under-estimation.
    """
    diff = y_pred - y_true
    scores = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    return float(np.mean(scores))


def health_score_distribution(scores: np.ndarray) -> dict:
    """Summarize a fleet health score distribution."""
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "std": float(np.std(scores)),
    }
