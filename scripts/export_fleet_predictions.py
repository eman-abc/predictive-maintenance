#!/usr/bin/env python
"""Rebuild fleet predictions parquet from saved Phase 3 models (no retraining)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.models.anomaly_detector import AnomalyDetector
from src.models.cmapss_eval import load_feature_columns
from src.models.cmapss_phase3 import ANOMALY_MAX_TRAIN_ROWS, ANOMALY_MIN_RUL_FIT, build_fleet_predictions
from src.models.failure_classifier import FailureClassifier
from src.models.lstm_model import LSTMModel
from src.models.rul_regressor import RULRegressor
from src.models.survival_model import SurvivalModel

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")


def _load_rul_model(dataset_id: str, winner: str):
    if winner == "lstm":
        path = MODELS_DIR / f"rul_lstm_{dataset_id}.pt"
        return LSTMModel.load(path)
    path = MODELS_DIR / f"rul_{winner}_{dataset_id}.pkl"
    return RULRegressor.load(path)


def export(dataset_id: str) -> Path:
    summary_path = ARTIFACTS_DIR / f"cmapss_{dataset_id}_phase3_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run train_cmapss_phase3.py --dataset {dataset_id} first."
        )
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    winner = summary["winner"]

    test_path = PROCESSED_DIR / f"cmapss_{dataset_id}_test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    test_df = pd.read_parquet(test_path)
    feature_cols = load_feature_columns(ARTIFACTS_DIR, dataset_id)
    rul_model = _load_rul_model(dataset_id, winner)

    clf_30_path = MODELS_DIR / f"failure_30_{dataset_id}.pkl"
    clf_72_path = MODELS_DIR / f"failure_72_{dataset_id}.pkl"
    if not clf_30_path.exists():
        raise FileNotFoundError(f"Missing {clf_30_path}")

    failure_clf_30 = FailureClassifier.load(clf_30_path)
    failure_clf_72 = (
        FailureClassifier.load(clf_72_path) if clf_72_path.exists() else None
    )

    anomaly_path = MODELS_DIR / f"anomaly_{dataset_id}.pkl"
    if anomaly_path.exists():
        anomaly_det = AnomalyDetector.load(anomaly_path)
    else:
        train_path = PROCESSED_DIR / f"cmapss_{dataset_id}_train.parquet"
        train_df = pd.read_parquet(train_path)
        anomaly_det = AnomalyDetector()
        anomaly_det.fit(
            train_df,
            feature_cols,
            min_rul=ANOMALY_MIN_RUL_FIT,
            max_rows=ANOMALY_MAX_TRAIN_ROWS,
        )
        anomaly_det.save(anomaly_path)
        print(f"Trained and saved {anomaly_path}")

    survival_path = MODELS_DIR / f"survival_{dataset_id}.pkl"
    survival_model = (
        SurvivalModel.load(survival_path) if survival_path.exists() else None
    )

    fleet = build_fleet_predictions(
        test_df,
        rul_model,
        failure_clf_30,
        feature_cols,
        model_name=winner,
        failure_clf_72=failure_clf_72,
        anomaly_detector=anomaly_det,
        survival_model=survival_model,
        dataset_id=dataset_id,
    )
    out = PROCESSED_DIR / f"cmapss_{dataset_id}_predictions.parquet"
    fleet.to_parquet(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fleet predictions parquet")
    parser.add_argument("--dataset", default="FD003")
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()
    for ds in args.datasets or [args.dataset]:
        path = export(ds)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
