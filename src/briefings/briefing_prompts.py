"""Prompt engineering templates for maintenance briefings."""

from __future__ import annotations

from typing import Any

# Short system prompt = fewer tokens processed per request.
SYSTEM_PROMPT = (
    "You write one short maintenance briefing for plant operators. "
    "Use plain language. Max 4 sentences. No bullet lists."
)


def build_briefing_prompt(
    asset_id: str,
    health_score: float,
    rul: float,
    failure_probability: float,
    sensor_summary: dict[str, Any] | None = None,
    *,
    max_sensors: int = 4,
) -> str:
    """Compact prompt tuned for fast local inference."""
    sensor_text = ""
    if sensor_summary:
        items = list(sensor_summary.items())[:max_sensors]
        sensor_text = " Sensors: " + ", ".join(f"{k}={v:.1f}" for k, v in items) + "."

    return (
        f"Asset {asset_id}. Health {health_score:.0f}/100. "
        f"RUL {rul:.0f} cycles. P(fail within 30 cycles) {failure_probability:.0%}."
        f"{sensor_text} "
        "Write one paragraph: condition, urgency (low/medium/high), and one recommended action."
    )


def build_instant_briefing(
    asset_id: str,
    health_score: float,
    rul: float,
    failure_probability: float,
    *,
    alert_level: str = "normal",
    recommended_action: str = "",
    anomaly_score: float = 0.0,
) -> str:
    """UC5-style operator briefing without calling the LLM (instant)."""
    action = recommended_action or "Continue routine monitoring."
    urgency = (
        "high"
        if alert_level == "critical"
        else "medium"
        if alert_level == "warning"
        else "low"
    )
    anom_note = ""
    if anomaly_score >= 55:
        anom_note = f" Anomaly score {anomaly_score:.0f}/100 vs healthy baseline."
    return (
        f"{asset_id} health score {health_score:.0f}/100 with predicted RUL {rul:.0f} cycles. "
        f"Failure probability within 30 cycles: {failure_probability:.0%}. "
        f"Urgency: {urgency}.{anom_note} {action}"
    )


def build_alert_summary_prompt(alerts: list[dict[str, Any]]) -> str:
    """Build a prompt summarizing multiple active alerts for a shift handover."""
    alert_lines = []
    for a in alerts:
        alert_lines.append(
            f"- [{a.get('level', 'unknown').upper()}] {a.get('asset_id')}: "
            f"{a.get('description', 'No description')}"
        )

    alerts_text = "\n".join(alert_lines) if alert_lines else "No active alerts."

    return f"""Summarize the following maintenance alerts for a shift handover briefing:

{alerts_text}

Provide reformat as:
1. Executive summary (2-3 sentences)
2. Priority actions ranked by urgency
3. Assets that can continue normal operation
"""

__all__ = [
    "SYSTEM_PROMPT",
    "build_briefing_prompt",
    "build_instant_briefing",
    "build_alert_summary_prompt",
]
