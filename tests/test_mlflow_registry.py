"""Tests for MLflow registry naming helpers."""

from src.models.mlflow_registry import registered_model_name


def test_registered_model_name_rul():
    assert registered_model_name("rul", "FD001", variant="gbm") == "cmapss_rul_gbm_FD001"


def test_registered_model_name_failure():
    assert registered_model_name("failure_30", "FD003") == "cmapss_failure_30_FD003"
