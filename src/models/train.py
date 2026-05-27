"""Training orchestrator with MLflow experiment logging."""

import os
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv

from src.ingestion.feature_engineer import FeatureEngineer
from src.models.failure_classifier import FailureClassifier
from src.models.rul_regressor import RULRegressor

load_dotenv()

MODELS_DIR = Path("models")
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed"))


def train_all(
    cmapss_path: str | None = None,
    ai4i_path: str | None = None,
) -> dict:
    """Train RUL regressor and failure classifier, log to MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance"))
    MODELS_DIR.mkdir(exist_ok=True)

    results = {}

    if cmapss_path and Path(cmapss_path).exists():
        df = pd.read_parquet(cmapss_path)
        feature_cols = [
            c for c in df.columns
            if c not in {"unit_id", "cycle", "rul", "failure"}
            and df[c].dtype in ["float64", "int64"]
        ]
        X = df[feature_cols].fillna(0)
        y_rul = df["rul"]

        with mlflow.start_run(run_name="rul_rf"):
            regressor = RULRegressor(model_type="rf")
            metrics = regressor.fit(X, y_rul)
            mlflow.log_metrics(metrics)
            regressor.save(MODELS_DIR / "rul_rf_model.pkl")
            results["rul"] = metrics

    if ai4i_path and Path(ai4i_path).exists():
        df = pd.read_parquet(ai4i_path)
        feature_cols = [
            c for c in df.columns
            if c not in {"UDI", "Product ID", "failure", "Machine failure", "rul"}
            and df[c].dtype in ["float64", "int64"]
        ]
        X = df[feature_cols].fillna(0)
        y_fail = df["failure"]

        with mlflow.start_run(run_name="failure_gbm"):
            classifier = FailureClassifier(model_type="gbm")
            metrics = classifier.fit(X, y_fail)
            mlflow.log_metrics(metrics)
            classifier.save(MODELS_DIR / "failure_clf_model.pkl")
            results["failure"] = metrics

    return results


def prepare_and_train(
    cmapss_raw_dir: str = "./data/raw/cmapss",
    ai4i_raw_dir: str = "./data/raw/ai4i",
) -> dict:
    """End-to-end: load raw data, engineer features, train models."""
    from src.ingestion.ai4i_loader import load_ai4i
    from src.ingestion.cmapss_loader import compute_train_rul, load_cmapss_train

    engineer = FeatureEngineer()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cmapss_path = PROCESSED_DIR / "cmapss_features.parquet"
    ai4i_path = PROCESSED_DIR / "ai4i_features.parquet"

    try:
        cmapss_df = load_cmapss_train(cmapss_raw_dir)
        cmapss_df = compute_train_rul(cmapss_df)
        cmapss_df = engineer.engineer_cmapss(cmapss_df)
        engineer.save_processed(cmapss_df, cmapss_path)
    except FileNotFoundError:
        cmapss_path = None

    try:
        ai4i_df = load_ai4i(ai4i_raw_dir)
        ai4i_df = engineer.engineer_ai4i(ai4i_df)
        engineer.save_processed(ai4i_df, ai4i_path)
    except FileNotFoundError:
        ai4i_path = None

    return train_all(
        str(cmapss_path) if cmapss_path and cmapss_path.exists() else None,
        str(ai4i_path) if ai4i_path and ai4i_path.exists() else None,
    )


if __name__ == "__main__":
    prepare_and_train()
