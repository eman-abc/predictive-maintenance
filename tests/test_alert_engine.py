"""Tests for alert threshold engine and generator."""

import pytest

from src.alerts.threshold_engine import AlertLevel, ThresholdEngine
from src.alerts.alert_generator import AlertGenerator


@pytest.fixture
def engine():
    return ThresholdEngine(
        rul_critical=10, rul_warning=30,
        prob_critical=0.85, prob_warning=0.60,
    )


def test_normal_assessment(engine):
    result = engine.assess("ENG-001", rul=80, failure_probability=0.1)
    assert result.alert_level == AlertLevel.NORMAL
    assert result.health_score > 50


def test_warning_assessment(engine):
    result = engine.assess("ENG-002", rul=25, failure_probability=0.3)
    assert result.alert_level == AlertLevel.WARNING


def test_critical_assessment(engine):
    result = engine.assess("ENG-003", rul=5, failure_probability=0.9)
    assert result.alert_level == AlertLevel.CRITICAL


def test_alert_generator_skips_normal(engine):
    gen = AlertGenerator()
    assessment = engine.assess("ENG-004", rul=100, failure_probability=0.05)
    assert gen.from_assessment(assessment) is None


def test_alert_generator_creates_alert(engine):
    gen = AlertGenerator()
    assessment = engine.assess("ENG-005", rul=5, failure_probability=0.9)
    alert = gen.from_assessment(assessment)
    assert alert is not None
    assert alert.level == AlertLevel.CRITICAL
    assert alert.asset_id == "ENG-005"
