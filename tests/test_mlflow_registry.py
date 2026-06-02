"""Tests for MLflow registry naming helpers."""

import os

from src.models.mlflow_registry import registered_model_name


def test_registered_model_name_rul_local(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "./mlruns")
    monkeypatch.delenv("MLFLOW_UC_CATALOG", raising=False)
    assert registered_model_name("rul", "FD001", variant="gbm") == "cmapss_rul_gbm_FD001"


def test_registered_model_name_failure_local(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "./mlruns")
    monkeypatch.delenv("MLFLOW_UC_CATALOG", raising=False)
    assert registered_model_name("failure_30", "FD003") == "cmapss_failure_30_FD003"


def test_registered_model_name_unity_catalog(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    monkeypatch.setenv("MLFLOW_UC_CATALOG", "main")
    monkeypatch.setenv("MLFLOW_UC_SCHEMA", "default")
    assert (
        registered_model_name("rul", "FD001", variant="gbm")
        == "main.default.cmapss_rul_gbm_FD001"
    )
