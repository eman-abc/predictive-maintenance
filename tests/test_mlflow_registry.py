"""Tests for MLflow registry naming helpers."""

import os

from src.models.mlflow_registry import registered_model_name


def test_registered_model_name_uc_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_USE_LEGACY_MODEL_REGISTRY", raising=False)
    assert (
        registered_model_name("rul", "FD001", variant="gbm")
        == "main.default.cmapss_rul_gbm_FD001"
    )


def test_registered_model_name_uc_custom(monkeypatch):
    monkeypatch.delenv("MLFLOW_USE_LEGACY_MODEL_REGISTRY", raising=False)
    monkeypatch.setenv("MLFLOW_UC_CATALOG", "workspace")
    monkeypatch.setenv("MLFLOW_UC_SCHEMA", "predictive_maintenance")
    assert (
        registered_model_name("failure_30", "FD003")
        == "workspace.predictive_maintenance.cmapss_failure_30_FD003"
    )


def test_registered_model_name_legacy(monkeypatch):
    monkeypatch.setenv("MLFLOW_USE_LEGACY_MODEL_REGISTRY", "1")
    assert registered_model_name("rul", "FD001", variant="gbm") == "cmapss_rul_gbm_FD001"
