"""Tests for CMAPSS training registry helpers."""

import json

from src.models.cmapss_phase3 import REGISTRY_PATH, update_training_registry


def test_update_training_registry_merges_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr("src.models.cmapss_phase3.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("src.models.cmapss_phase3.REGISTRY_PATH", tmp_path / "registry.json")

    summary1 = {
        "dataset_id": "FD001",
        "winner": "gbm",
        "skip_cox": False,
        "test_metrics": {"rmse": 12.0, "rul_score": 250.0},
        "cox_test_metrics": {"rmse": 13.5, "rul_score": 265.0},
        "predictions_path": "data/processed/cmapss_FD001_predictions.parquet",
    }
    update_training_registry(summary1, mlflow_run_id="run-1")
    summary2 = {
        "dataset_id": "FD003",
        "winner": "rf",
        "test_metrics": {"rmse": 14.0, "rul_score": 300.0},
        "predictions_path": "data/processed/cmapss_FD003_predictions.parquet",
    }
    update_training_registry(summary2, mlflow_run_id="run-2")

    data = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert set(data["datasets"]) == {"FD001", "FD003"}
    assert data["datasets"]["FD001"]["winner"] == "gbm"
    assert data["datasets"]["FD001"]["mlflow_run_id"] == "run-1"
    assert data["datasets"]["FD001"]["test_cox_nasa_score"] == 265.0
    assert data["datasets"]["FD001"]["skip_cox"] is False
    assert data["pipeline"] == "cmapss_phase3"
