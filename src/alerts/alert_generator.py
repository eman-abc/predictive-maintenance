"""Alert payload construction and dispatch."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.alerts.threshold_engine import AlertLevel, RiskAssessment


@dataclass
class Alert:
    alert_id: str
    asset_id: str
    level: AlertLevel
    title: str
    description: str
    health_score: float
    rul: float
    failure_probability: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["level"] = self.level.value
        return payload


class AlertGenerator:
    """Build structured alert payloads from risk assessments."""

    def __init__(self):
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"ALT-{self._counter:06d}"

    def from_assessment(self, assessment: RiskAssessment) -> Alert | None:
        """Create an alert if the assessment exceeds warning threshold."""
        if assessment.alert_level == AlertLevel.NORMAL:
            return None

        title = (
            "Critical Maintenance Alert"
            if assessment.alert_level == AlertLevel.CRITICAL
            else "Maintenance Warning"
        )

        return Alert(
            alert_id=self._next_id(),
            asset_id=assessment.asset_id,
            level=assessment.alert_level,
            title=title,
            description=assessment.message,
            health_score=assessment.health_score,
            rul=assessment.rul,
            failure_probability=assessment.failure_probability,
        )

    def batch_from_assessments(
        self, assessments: list[RiskAssessment]
    ) -> list[Alert]:
        return [
            alert
            for a in assessments
            if (alert := self.from_assessment(a)) is not None
        ]
