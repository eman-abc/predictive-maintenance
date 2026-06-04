"""Tests for escalation tier → CMMS priority / SLA mapping."""

from src.alerts.cmms_routing import map_escalation
from src.alerts.threshold_engine import AlertLevel


def test_l2_critical_sla():
    r = map_escalation("L2-Critical")
    assert r.cmms_priority == "P1"
    assert r.sla_response_hours == 4
    assert "4h" in r.sla_label


def test_l1_warning_sla():
    r = map_escalation("L1-Warning")
    assert r.cmms_priority == "P2"
    assert r.sla_response_hours == 72
    assert "72h" in r.sla_label


def test_fallback_from_alert_level():
    r = map_escalation(None, alert_level=AlertLevel.CRITICAL)
    assert r.escalation_tier == "L2-Critical"
