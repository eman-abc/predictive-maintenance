"""CMAPSS survival-analysis helpers (Cox PH — lifelines)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.cmapss_eval import evaluate_rul, last_cycle_per_unit, rul_validation_frame


def add_survival_columns(df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
    """
    Per-row survival format: duration = operating cycle, event = 1 on engine failure.

    Train: failure observed on each unit's terminal cycle.
    Test / val scoring rows: event = 0 (censored at last observed cycle).
    """
    out = df.copy()
    terminal = df.groupby("unit_id")["cycle"].transform("max")
    out["duration"] = out["cycle"].astype(float)
    if is_train:
        out["event"] = (out["cycle"] == terminal).astype(int)
    else:
        out["event"] = 0
    return out


def select_cox_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    max_features: int = 40,
) -> list[str]:
    """Reduce collinearity risk by keeping highest-variance engineered features."""
    present = [c for c in feature_cols if c in df.columns]
    if len(present) <= max_features:
        return present
    var = df[present].fillna(0).var().sort_values(ascending=False)
    return var.head(max_features).index.tolist()


def evaluate_cox_rul(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    concordance: float | None = None,
) -> dict[str, float]:
    """NASA/RMSE on RUL proxy plus optional lifelines concordance."""
    metrics = evaluate_rul(y_true, y_pred)
    if concordance is not None and np.isfinite(concordance):
        metrics["concordance"] = float(concordance)
    return metrics
