"""CMMS and Databricks API routes."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("OLLAMA_PRELOAD_ON_STARTUP", "false")
    return TestClient(app)


def test_cmms_databricks_status_unconfigured(client: TestClient, monkeypatch):
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("CMMS_LOG_TO_DATABRICKS", "false")
    resp = client.get("/cmms/databricks/status")
    assert resp.status_code == 200
    assert resp.json().get("configured") is False


def test_auto_dispatch_skipped_when_logging_off(client: TestClient, monkeypatch):
    monkeypatch.setenv("CMMS_LOG_TO_DATABRICKS", "false")
    resp = client.post(
        "/cmms/auto-dispatch",
        json={"dataset_id": "FD001", "levels": ["critical"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "skipped"


@patch("src.api.main.auto_dispatch_critical")
def test_auto_dispatch_endpoint(mock_dispatch, client: TestClient):
    mock_dispatch.return_value = {
        "status": "ok",
        "table": "workspace.cmapss.cmms_work_orders_auto",
        "dispatched_count": 2,
        "skipped_count": 0,
        "skipped_assets": [],
        "results": [],
    }
    resp = client.post(
        "/cmms/auto-dispatch",
        json={"dataset_id": "FD003", "levels": ["critical"]},
    )
    assert resp.status_code == 200
    assert resp.json()["dispatched_count"] == 2
    mock_dispatch.assert_called_once_with("FD003", levels=["critical"])


def test_workorders_submit_calls_cmms(client: TestClient, monkeypatch):
    monkeypatch.setattr(
        "src.api.main.alerts_service.build_alerts_for_cmms",
        lambda dataset_id, levels=None: [],
    )
    resp = client.post(
        "/cmms/workorders",
        json={"dataset_id": "FD001", "levels": ["critical"]},
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_recent_work_orders_unconfigured(client: TestClient, monkeypatch):
    monkeypatch.setenv("CMMS_LOG_TO_DATABRICKS", "false")
    resp = client.get("/cmms/workorders/recent")
    assert resp.status_code == 200
    assert resp.json()["configured"] is False
    assert resp.json()["rows"] == []


def test_recent_auto_work_orders_unconfigured(client: TestClient, monkeypatch):
    monkeypatch.setenv("CMMS_LOG_TO_DATABRICKS", "false")
    resp = client.get("/cmms/workorders/recent/auto")
    assert resp.status_code == 200
    assert resp.json()["configured"] is False
