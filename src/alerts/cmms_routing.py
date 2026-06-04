"""Map escalation tiers to CMMS priority and SLA (UC5 Component C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.alerts.threshold_engine import AlertLevel


@dataclass(frozen=True)
class CmmsRouting:
    escalation_tier: str
    cmms_priority: str
    sla_response_hours: int
    sla_label: str
    priority_rank: int  # 1 = highest


_TIER_MAP: dict[str, CmmsRouting] = {
    "L2-Critical": CmmsRouting(
        escalation_tier="L2-Critical",
        cmms_priority="P1",
        sla_response_hours=4,
        sla_label="4h response SLA",
        priority_rank=1,
    ),
    "L1-Warning": CmmsRouting(
        escalation_tier="L1-Warning",
        cmms_priority="P2",
        sla_response_hours=72,
        sla_label="72h response SLA",
        priority_rank=2,
    ),
    "L0-Normal": CmmsRouting(
        escalation_tier="L0-Normal",
        cmms_priority="P3",
        sla_response_hours=168,
        sla_label="168h routine SLA",
        priority_rank=3,
    ),
}


def map_escalation(
    escalation_tier: str | None,
    *,
    alert_level: str | AlertLevel | None = None,
) -> CmmsRouting:
    """
    Resolve CMMS priority / SLA from escalation_tier (preferred) or alert_level.
    """
    tier = (escalation_tier or "").strip()
    if tier in _TIER_MAP:
        return _TIER_MAP[tier]

    level = alert_level.value if isinstance(alert_level, AlertLevel) else str(alert_level or "")
    if level == AlertLevel.CRITICAL.value:
        return _TIER_MAP["L2-Critical"]
    if level == AlertLevel.WARNING.value:
        return _TIER_MAP["L1-Warning"]
    return _TIER_MAP["L0-Normal"]


def routing_to_payload_fields(routing: CmmsRouting) -> dict[str, Any]:
    return {
        "escalation_tier": routing.escalation_tier,
        "cmms_priority": routing.cmms_priority,
        "sla_response_hours": routing.sla_response_hours,
        "sla_label": routing.sla_label,
    }
