"""Build UC5-complete alert fields from fleet / assessment rows."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.alerts.alert_generator import Alert
from src.alerts.threshold_engine import AlertLevel, RiskAssessment

# Representative sensors for operator-facing alerts (CMAPSS keeps 14–21 per FD).
DEFAULT_SNAPSHOT_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_15",
    "sensor_20",
]


def snapshot_sensor_readings(
    row: pd.Series,
    sensors: list[str] | None = None,
) -> dict[str, float]:
    """Current sensor values from one cycle row (typically last test cycle)."""
    cols = sensors or [
        c for c in row.index if str(c).startswith("sensor_")
    ]
    out: dict[str, float] = {}
    for col in cols:
        if col in row.index and pd.notna(row[col]):
            out[str(col)] = float(row[col])
    return out


def recommended_maintenance_action(
    level: AlertLevel,
    rul: float,
    failure_probability: float,
    *,
    anomaly_score: float = 0.0,
) -> str:
    """Plain-language maintenance action for operators and CMMS work orders."""
    if level == AlertLevel.CRITICAL:
        if rul <= 5:
            return (
                "Emergency planning: stop-at-next-window inspection of core engine, "
                "verify lubrication and vibration; prepare swap for critical components."
            )
        return (
            "Immediate inspection (borescope + oil analysis + vibration survey); "
            "schedule component service before RUL reaches zero."
        )
    if level == AlertLevel.WARNING:
        base = (
            "Schedule planned maintenance within the predicted RUL window: "
            "bearing temperature check, oil sampling, and align spare parts."
        )
        if anomaly_score >= 60:
            return base + " Prioritize due to elevated anomaly score vs healthy baseline."
        return base
    return "Continue routine monitoring; no maintenance dispatch required."


def enrich_assessment(
    assessment: RiskAssessment,
    *,
    sensor_readings: dict[str, float],
    anomaly_score: float = 0.0,
) -> RiskAssessment:
    """Attach UC5 fields to a risk assessment (mutates message with action)."""
    action = recommended_maintenance_action(
        assessment.alert_level,
        assessment.rul,
        assessment.failure_probability,
        anomaly_score=anomaly_score,
    )
    assessment.message = f"{assessment.message} | Action: {action}"
    return assessment


def fleet_row_to_alert(row: pd.Series, alert_id: str | None = None) -> Alert | None:
    """Convert a predictions-parquet row to a structured Alert (None if normal)."""
    level = AlertLevel(str(row["alert_level"]))
    if level == AlertLevel.NORMAL:
        return None

    sensors = {}
    if "sensor_readings_json" in row.index and pd.notna(row["sensor_readings_json"]):
        sensors = json.loads(str(row["sensor_readings_json"]))
    elif "sensor_readings" in row.index:
        raw = row["sensor_readings"]
        sensors = raw if isinstance(raw, dict) else json.loads(str(raw))

    risk = float(row.get("risk_score", row.get("health_score", 0)))
    rul = float(row.get("time_to_failure_cycles", row.get("rul_pred", 0)))
    action = str(
        row.get(
            "recommended_action",
            recommended_maintenance_action(
                level,
                rul,
                float(row.get("failure_prob_30", row.get("failure_prob", 0))),
                anomaly_score=float(row.get("anomaly_score", 0)),
            ),
        )
    )

    return Alert(
        alert_id=alert_id or f"ALT-{int(row['unit_id']):06d}",
        asset_id=str(row["asset_id"]),
        level=level,
        title=f"{level.value.title()} Maintenance Alert — {row['asset_id']}",
        description=str(row.get("alert_message", row.get("message", ""))),
        health_score=risk,
        rul=rul,
        failure_probability=float(row.get("failure_prob_30", row.get("failure_prob", 0))),
        metadata={
            "risk_score": risk,
            "time_to_failure_cycles": rul,
            "recommended_action": action,
            "sensor_readings": sensors,
            "anomaly_score": float(row.get("anomaly_score", 0)),
            "failure_prob_72": float(row.get("failure_prob_72", 0)),
            "escalation_tier": str(row.get("escalation_tier", "")),
        },
    )
