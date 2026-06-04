"""Active alerts from fleet predictions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.alerts.alert_payload import fleet_row_to_alert
from src.services.fleet_service import load_fleet_predictions


def list_alert_rows(
    dataset_id: str,
    *,
    levels: list[str] | None = None,
) -> list[dict[str, Any]]:
    fleet = load_fleet_predictions(dataset_id)
    if fleet is None:
        return []
    levels = levels or ["critical", "warning"]
    filtered = fleet[fleet["alert_level"].isin(levels)]
    return filtered.replace({float("inf"): None}).to_dict(orient="records")


def build_alerts_for_cmms(
    dataset_id: str,
    *,
    levels: list[str] | None = None,
) -> list[tuple[Any, str]]:
    """Return (Alert, alert_id) pairs for CMMS submission."""
    fleet = load_fleet_predictions(dataset_id)
    if fleet is None:
        return []
    levels = levels or ["critical", "warning"]
    filtered = fleet[fleet["alert_level"].isin(levels)]
    out = []
    for i, (_, row) in enumerate(filtered.iterrows(), start=1):
        alert_id = f"ALT-{i:06d}"
        alert = fleet_row_to_alert(row, alert_id=alert_id)
        if alert:
            out.append((alert, alert_id))
    return out
