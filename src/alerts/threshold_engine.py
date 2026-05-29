"""Risk scoring and threshold logic for maintenance alerts."""

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class AlertLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    asset_id: str
    rul: float
    failure_probability: float
    health_score: float
    alert_level: AlertLevel
    message: str
    recommended_action: str = ""
    risk_score: float = 0.0
    time_to_failure_cycles: float = 0.0


class ThresholdEngine:
    """Evaluate RUL and failure probability against configurable thresholds."""

    def __init__(
        self,
        rul_critical: float | None = None,
        rul_warning: float | None = None,
        prob_critical: float | None = None,
        prob_warning: float | None = None,
    ):
        self.rul_critical = float(rul_critical or os.getenv("ALERT_RUL_CRITICAL", 10))
        self.rul_warning = float(rul_warning or os.getenv("ALERT_RUL_WARNING", 30))
        self.prob_critical = float(
            prob_critical or os.getenv("ALERT_FAILURE_PROB_CRITICAL", 0.85)
        )
        self.prob_warning = float(
            prob_warning or os.getenv("ALERT_FAILURE_PROB_WARNING", 0.60)
        )

    def compute_health_score(self, rul: float, failure_prob: float, max_rul: float = 125) -> float:
        """Health score 0-100 combining RUL headroom and failure risk."""
        rul_component = min(rul / max_rul, 1.0) * 100
        risk_component = (1 - failure_prob) * 100
        return round(0.6 * rul_component + 0.4 * risk_component, 1)

    def assess(
        self,
        asset_id: str,
        rul: float,
        failure_probability: float,
        *,
        anomaly_score: float = 0.0,
        sensor_readings: dict[str, float] | None = None,
    ) -> RiskAssessment:
        """Determine alert level and generate human-readable message."""
        from src.alerts.alert_payload import recommended_maintenance_action

        health = self.compute_health_score(rul, failure_probability)
        sensor_readings = sensor_readings or {}

        if (
            rul <= self.rul_critical
            or failure_probability >= self.prob_critical
            or (anomaly_score >= 75 and rul <= self.rul_warning)
        ):
            level = AlertLevel.CRITICAL
            message = (
                f"Asset {asset_id}: immediate maintenance required "
                f"(RUL={rul:.0f} cycles, P(fail≤30)={failure_probability:.0%}, "
                f"anomaly={anomaly_score:.0f})"
            )
        elif (
            rul <= self.rul_warning
            or failure_probability >= self.prob_warning
            or anomaly_score >= 55
        ):
            level = AlertLevel.WARNING
            message = (
                f"Asset {asset_id}: schedule maintenance soon "
                f"(RUL={rul:.0f} cycles, P(fail≤30)={failure_probability:.0%}, "
                f"anomaly={anomaly_score:.0f})"
            )
        else:
            level = AlertLevel.NORMAL
            message = f"Asset {asset_id}: operating normally (health={health:.0f}%)"

        action = recommended_maintenance_action(
            level, rul, failure_probability, anomaly_score=anomaly_score
        )

        return RiskAssessment(
            asset_id=asset_id,
            rul=rul,
            failure_probability=failure_probability,
            health_score=health,
            alert_level=level,
            message=message,
            recommended_action=action,
            risk_score=health,
            time_to_failure_cycles=rul,
        )
