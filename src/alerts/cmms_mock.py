"""Mock CMMS API integration for work order creation."""

import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.alerts.alert_generator import Alert
from src.alerts.cmms_routing import map_escalation, routing_to_payload_fields

load_dotenv()


class CMMSClient:
    """Mock client that posts maintenance work orders to a CMMS endpoint."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or os.getenv("CMMS_API_URL", "http://localhost:8080/api/workorders")
        self.api_key = api_key or os.getenv("CMMS_API_KEY", "")

    def create_work_order(
        self,
        alert: Alert,
        *,
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit a work order derived from a UC5-complete alert payload."""
        meta = alert.metadata or {}
        if dataset_id:
            meta = {**meta, "dataset_id": dataset_id}
            alert.metadata = meta
        routing = map_escalation(
            meta.get("escalation_tier"),
            alert_level=alert.level,
        )
        payload = {
            "asset_id": alert.asset_id,
            "alert_id": alert.alert_id,
            **routing_to_payload_fields(routing),
            "priority": routing.cmms_priority,
            "priority_legacy": "high" if alert.level.value == "critical" else "medium",
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

        result: dict[str, Any]
        try:
            response = requests.post(
                self.base_url, json=payload, headers=headers, timeout=10
            )
            response.raise_for_status()
            result = {"status": "submitted", "response": response.json()}
        except requests.RequestException:
            result = {"status": "mock_logged", "payload": payload}

        result = self._maybe_log_databricks(
            alert, submit_status=result["status"], payload=payload, result=result, dataset_id=dataset_id
        )
        return result

    def _maybe_log_databricks(
        self,
        alert: Alert,
        *,
        submit_status: str,
        payload: dict[str, Any],
        result: dict[str, Any],
        dataset_id: str | None,
    ) -> dict[str, Any]:
        try:
            from src.alerts.cmms_databricks import insert_work_order, is_databricks_logging_configured

            if not is_databricks_logging_configured():
                return result
            db = insert_work_order(
                alert,
                submit_status=submit_status,
                payload=payload,
                dataset_id=dataset_id,
                cmms_response=result,
            )
            result["databricks"] = db
        except Exception as exc:
            result["databricks"] = {"status": "error", "message": str(exc)}
        return result
