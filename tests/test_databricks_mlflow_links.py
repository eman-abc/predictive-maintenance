"""Tests for Databricks MLflow link fallback (no API)."""

from __future__ import annotations

import os

import pytest

from dashboard import mlflow_links as links


@pytest.fixture
def sample_registry():
    return {
        "mlflow_experiment_id": "2311620089394370",
        "datasets": {
            "FD001": {
                "mlflow_run_id": "1ea139288b41444a9e98292cb22914f0",
                "mlflow_run_name": "FD001_phase3_summary",
                "winner": "gbm",
                "test_rmse": 23.3,
                "test_nasa_score": 10.3,
            }
        },
    }


def test_run_url_requires_host(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://dbc-test.cloud.databricks.com")
    url = links.run_url("2311620089394370", "abc123")
    assert url == "https://dbc-test.cloud.databricks.com/ml/experiments/2311620089394370/runs/abc123/overview"


def test_registry_fallback_without_token(monkeypatch, sample_registry):
    monkeypatch.setenv("DATABRICKS_HOST", "https://dbc-test.cloud.databricks.com")
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("MLFLOW_PKL_BUNDLE_RUN_ID", raising=False)

    assert links.can_build_registry_links(sample_registry)
    data = links.collect_databricks_run_links(sample_registry)
    assert data["source"] == "registry"
    assert data["registry_only"] is True
    assert len(data["phase3_rows"]) == 1
    assert "1ea139288b41444a9e98292cb22914f0" in data["phase3_rows"][0]["run_url"]


def test_env_experiment_id_overrides_registry(monkeypatch, sample_registry):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "999")
    assert links.resolve_experiment_id(sample_registry) == "999"


def test_pkl_bundle_env_link(monkeypatch, sample_registry):
    monkeypatch.setenv("DATABRICKS_HOST", "https://dbc-test.cloud.databricks.com")
    monkeypatch.setenv("MLFLOW_PKL_BUNDLE_RUN_ID", "d33506a96f6e41e1a4144ca50b701b2f")
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    data = links.collect_databricks_run_links(sample_registry)
    assert len(data["bundle_rows"]) == 1
    assert "d33506a96f6e41e1a4144ca50b701b2f" in data["bundle_rows"][0]["artifacts_url"]
