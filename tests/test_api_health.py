"""FastAPI health endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "datasets_available" in data


def test_datasets_endpoint(client: TestClient):
    resp = client.get("/datasets")
    assert resp.status_code == 200
    assert "datasets" in resp.json()


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/processed/cmapss_FD001_predictions.parquet").exists(),
    reason="FD001 predictions not on disk",
)
def test_fleet_fd001(client: TestClient):
    resp = client.get("/fleet", params={"dataset": "FD001"})
    assert resp.status_code == 200
    assert resp.json()["count"] > 0
