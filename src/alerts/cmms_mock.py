"""Mock CMMS API integration for work order creation."""

import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.alerts.alert_generator import Alert

load_dotenv()


class CMMSClient:
    """Mock client that posts maintenance work orders to a CMMS endpoint."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or os.getenv("CMMS_API_URL", "http://localhost:8080/api/workorders")
        self.api_key = api_key or os.getenv("CMMS_API_KEY", "")

    def create_work_order(self, alert: Alert) -> dict[str, Any]:
        """Submit a work order derived from a UC5-complete alert payload."""
        meta = alert.metadata or {}
        payload = {
            "asset_id": alert.asset_id,
            "alert_id": alert.alert_id,
            "priority": "high" if alert.level.value == "critical" else "medium",
            "escalation_tier": meta.get("escalation_tier", alert.level.value),
            "description": alert.description,
            "recommended_action": meta.get(
                "recommended_action", alert.description
            ),
            "risk_score": meta.get("risk_score", alert.health_score),
            "time_to_failure_cycles": meta.get("time_to_failure_cycles", alert.rul),
            "failure_probability_30": alert.failure_probability,
            "failure_probability_72": meta.get("failure_prob_72"),
            "anomaly_score": meta.get("anomaly_score"),
            "sensor_readings": meta.get("sensor_readings", {}),
            "health_score": alert.health_score,
            "predicted_rul": alert.rul,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            response = requests.post(
                self.base_url, json=payload, headers=headers, timeout=10
            )
            response.raise_for_status()
            return {"status": "submitted", "response": response.json()}
        except requests.RequestException:
            return {"status": "mock_logged", "payload": payload}
