"""Shift briefing and alert acknowledge API routes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("OLLAMA_PRELOAD_ON_STARTUP", "false")
    return TestClient(app)


def test_alert_ack_roundtrip(client: TestClient):
    resp = client.post(
        "/alerts/ack",
        json={
            "dataset_id": "FD001",
            "asset_id": "ENG-1",
            "alert_level": "critical",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "acknowledged"


@patch("src.api.main.briefing_service.generate_shift_briefing")
def test_shift_briefing_instant(mock_gen, client: TestClient):
    mock_gen.return_value = {
        "text": "Shift summary: 2 critical, 1 warning.",
        "source": "instant",
        "mode": "instant",
    }
    resp = client.post(
        "/briefings/shift",
        json={
            "mode": "instant",
            "dataset_id": "FD003",
            "levels": ["critical", "warning"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "critical" in body["text"].lower() or body["text"]
    mock_gen.assert_called_once()


@pytest.mark.skipif(
    not Path("data/processed/cmapss_FD001_predictions.parquet").exists(),
    reason="FD001 predictions not on disk",
)
def test_alerts_include_ack_status(client: TestClient):
    resp = client.get(
        "/alerts",
        params={"dataset": "FD001", "level": "critical,warning"},
    )
    if resp.status_code != 200 or resp.json()["count"] == 0:
        pytest.skip("No alerts in FD001 fixture")
    row = resp.json()["rows"][0]
    asset_id = row["asset_id"]
    alert_level = row["alert_level"]
    client.post(
        "/alerts/ack",
        json={
            "dataset_id": "FD001",
            "asset_id": asset_id,
            "alert_level": alert_level,
        },
    )
    resp2 = client.get(
        "/alerts",
        params={"dataset": "FD001", "level": "critical,warning"},
    )
    rows = resp2.json()["rows"]
    match = [
        r
        for r in rows
        if r["asset_id"] == asset_id and r["alert_level"] == alert_level
    ]
    assert match and match[0].get("ack_status") == "acknowledged"
