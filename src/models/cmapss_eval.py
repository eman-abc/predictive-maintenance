"""CMAPSS evaluation utilities (Phase 3 — last-cycle benchmark protocol)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.metrics import rul_score

LABEL_COLS = {
    "unit_id",
    "cycle",
    "rul",
    "failure_30",
    "failure_72",
    "op_cluster",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
}


def load_feature_columns(artifacts_dir: Path, dataset_id: str) -> list[str]:
    path = artifacts_dir / f"cmapss_{dataset_id}_feature_columns.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per engine at max(cycle) — official CMAPSS test scoring point."""
    idx = df.groupby("unit_id")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit_id").reset_index(drop=True)


def rul_validation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows for RUL model selection on training trajectories.

    Excludes each engine's terminal cycle: on train data RUL is always 0 at
    end-of-life, so last-cycle-only validation trivially favors models that
    predict zero (especially LSTM).
    """
    terminal_idx = df.groupby("unit_id")["cycle"].idxmax()
    return (
        df.drop(index=terminal_idx)
        .sort_values(["unit_id", "cycle"])
        .reset_index(drop=True)
    )


def prepare_xy(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "rul"
) -> tuple[pd.DataFrame, pd.Series]:
    frame = last_cycle_per_unit(df)
    X = frame[feature_cols].fillna(0)
    y = frame[target_col]
    return X, y


def prepare_xy_validation(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "rul"
) -> tuple[pd.DataFrame, pd.Series]:
    """Features and targets for model selection (non-terminal cycles only)."""
    frame = rul_validation_frame(df)
    X = frame[feature_cols].fillna(0)
    y = frame[target_col]
    return X, y


def evaluate_rul(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rul_score": rul_score(y_true, y_pred),
    }


def evaluate_failure_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    eval_protocol: str,
) -> dict[str, float | int | str]:
    """
    Metrics for failure-within-horizon classifiers (UC5 Component B).

    eval_protocol documents how rows were chosen (e.g. non-terminal val cycles
    vs last test cycle for operational alerting).
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    out: dict[str, float | int | str] = {
        "eval_protocol": eval_protocol,
        "n_samples": int(len(y_true)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return out


def rank_models(
    results: dict[str, dict[str, float]],
    *,
    primary_metric: str = "rul_score",
    tie_priority: dict[str, int] | None = None,
) -> str:
    """
    Select winner by lowest NASA score; tie-break with lower RMSE then simpler model.
    """
    tie_priority = tie_priority or {"rf": 0, "gbm": 1, "lstm": 2}

    def sort_key(name: str) -> tuple:
        m = results[name]
        score = m.get(primary_metric, float("inf"))
        if score is None or not np.isfinite(score):
            score = float("inf")
        rmse = m.get("rmse", float("inf"))
        if rmse is None or not np.isfinite(rmse):
            rmse = float("inf")
        # Perfect val scores on train trajectories usually mean EOL leakage.
        if score == 0.0 and rmse == 0.0:
            score = float("inf")
        return (score, rmse, tie_priority.get(name, 99))

    return min(results.keys(), key=sort_key)
