"""Tests for UC5 alert payload helpers."""

import json

import pandas as pd

from src.alerts.alert_payload import fleet_row_to_alert, snapshot_sensor_readings
from src.alerts.threshold_engine import AlertLevel


def test_snapshot_sensor_readings():
    row = pd.Series({"sensor_2": 1.5, "sensor_3": 2.0, "rul": 10})
    snap = snapshot_sensor_readings(row)
    assert snap == {"sensor_2": 1.5, "sensor_3": 2.0}


def test_fleet_row_to_alert():
    row = pd.Series(
        {
            "unit_id": 1,
            "asset_id": "ENG-001",
            "alert_level": "critical",
            "risk_score": 12.0,
            "health_score": 12.0,
            "time_to_failure_cycles": 5.0,
            "rul_pred": 5.0,
            "failure_prob_30": 0.9,
            "failure_prob_72": 0.95,
            "failure_prob": 0.9,
            "anomaly_score": 80.0,
            "recommended_action": "Inspect now",
            "alert_message": "Critical",
            "escalation_tier": "L2-Critical",
            "sensor_readings_json": json.dumps({"sensor_2": 1.0}),
        }
    )
    alert = fleet_row_to_alert(row)
    assert alert is not None
    assert alert.level == AlertLevel.CRITICAL
    assert alert.metadata["sensor_readings"]["sensor_2"] == 1.0
