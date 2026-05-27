"""Optional Cox Proportional Hazards survival model."""

from pathlib import Path

import pandas as pd


class SurvivalModel:
    """Cox PH wrapper using lifelines (optional dependency)."""

    def __init__(self):
        try:
            from lifelines import CoxPHFitter
        except ImportError as exc:
            raise ImportError(
                "lifelines is required for survival analysis. "
                "Install with: pip install lifelines"
            ) from exc
        self.model = CoxPHFitter()
        self.feature_cols: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "cycle",
        event_col: str = "event",
        feature_cols: list[str] | None = None,
    ) -> dict:
        """Fit Cox PH model on survival-formatted data."""
        self.feature_cols = feature_cols or [
            c for c in df.columns if c.startswith("sensor_")
        ]
        cols = self.feature_cols + [duration_col, event_col]
        self.model.fit(df[cols], duration_col=duration_col, event_col=event_col)
        return {"concordance": float(self.model.concordance_index_)}

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        return self.model.predict_median(df[self.feature_cols])

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SurvivalModel":
        import joblib

        instance = cls()
        data = joblib.load(path)
        instance.model = data["model"]
        instance.feature_cols = data["feature_cols"]
        return instance
