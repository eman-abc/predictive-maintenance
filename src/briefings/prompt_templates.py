"""Prompt engineering templates for maintenance briefings."""

from typing import Any


SYSTEM_PROMPT = (
    "You are an industrial maintenance engineer assistant. "
    "Provide concise, actionable maintenance briefings based on sensor data "
    "and predictive model outputs. Use plain language suitable for plant operators."
)


def build_briefing_prompt(
    asset_id: str,
    health_score: float,
    rul: float,
    failure_probability: float,
    sensor_summary: dict[str, Any] | None = None,
) -> str:
    """Build a prompt for a per-asset maintenance briefing."""
    sensor_text = ""
    if sensor_summary:
        sensor_lines = [f"  - {k}: {v}" for k, v in sensor_summary.items()]
        sensor_text = "\nSensor readings:\n" + "\n".join(sensor_lines)

    return f"""Generate a maintenance briefing for the following asset:

Asset ID: {asset_id}
Health Score: {health_score}/100
Predicted RUL: {rul:.0f} cycles
Failure Probability: {failure_probability:.1%}
{sensor_text}

Include:
1. Current condition assessment
2. Recommended actions (if any)
3. Estimated urgency (low/medium/high)
4. Suggested inspection checklist items
"""


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
