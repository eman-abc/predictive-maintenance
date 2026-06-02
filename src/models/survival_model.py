"""Cox Proportional Hazards survival model (lifelines) for RUL and survival curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _survival_at_step(surv_series: pd.Series, t: float) -> float:
    """Step-function survival S(t) from a lifelines predict_survival_function column."""
    if t <= 0:
        return 1.0
    times = surv_series.index.to_numpy(dtype=float)
    if len(times) == 0:
        return 1.0
    pos = int(np.searchsorted(times, t, side="right")) - 1
    if pos < 0:
        return 1.0
    return float(surv_series.iloc[pos])


class SurvivalModel:
    """Cox PH wrapper: median remaining life + survival probabilities over future cycles."""

    def __init__(self, penalizer: float = 0.05):
        self.penalizer = penalizer
        self.model = None
        self.feature_cols: list[str] = []
        self._fitted = False

    def _ensure_lifelines(self):
        if self.model is None:
            from lifelines import CoxPHFitter

            self.model = CoxPHFitter(penalizer=self.penalizer)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        *,
        duration_col: str = "duration",
        event_col: str = "event",
        max_rows: int | None = None,
    ) -> dict[str, float]:
        """Fit Cox PH on survival-formatted rows."""
        self._ensure_lifelines()
        self.feature_cols = list(feature_cols)
        fit_df = df
        if max_rows is not None and len(fit_df) > max_rows:
            fit_df = fit_df.sample(n=max_rows, random_state=42)

        cols = self.feature_cols + [duration_col, event_col]
        X = fit_df[cols].fillna(0).astype(float)
        self.model.fit(X, duration_col=duration_col, event_col=event_col)
        self._fitted = True
        return {"concordance": float(self.model.concordance_index_)}

    def predict_median_lifetime(self, df: pd.DataFrame) -> np.ndarray:
        """Median total survival time (cycles from start) per row."""
        self._ensure_fitted()
        X = df[self.feature_cols].fillna(0)
        return np.asarray(self.model.predict_median(X), dtype=float)

    def predict_remaining_rul(
        self, df: pd.DataFrame, current_cycles: np.ndarray | pd.Series
    ) -> np.ndarray:
        """RUL proxy: median lifetime minus current cycle, clipped at zero."""
        median_life = self.predict_median_lifetime(df)
        current = np.asarray(current_cycles, dtype=float)
        return np.maximum(median_life - current, 0.0)

    def predict_survival_probability(
        self,
        df: pd.DataFrame,
        current_cycles: np.ndarray | pd.Series,
        *,
        horizon: float,
    ) -> np.ndarray:
        """
        P(engine survives at least ``horizon`` more cycles from current cycle).

        Uses S(t+horizon) / S(t) from the fitted Cox survival function.
        """
        self._ensure_fitted()
        X = df[self.feature_cols].fillna(0)
        current = np.asarray(current_cycles, dtype=float)
        surv_df = self.model.predict_survival_function(X)
        probs = np.zeros(len(X), dtype=float)
        for i in range(len(X)):
            series = surv_df.iloc[:, i]
            t0 = float(current[i])
            s0 = _survival_at_step(series, t0)
            s1 = _survival_at_step(series, t0 + horizon)
            probs[i] = s1 / s0 if s0 > 1e-9 else 0.0
        return np.clip(probs, 0.0, 1.0)

    def survival_curve(
        self,
        row: pd.Series | pd.DataFrame,
        *,
        current_cycle: float,
        max_future: int = 80,
    ) -> pd.DataFrame:
        """Future cycles vs conditional survival probability from current_cycle."""
        self._ensure_fitted()
        if isinstance(row, pd.Series):
            frame = row.to_frame().T
        else:
            frame = row
        X = frame[self.feature_cols].fillna(0)
        series = self.model.predict_survival_function(X).iloc[:, 0]
        t0 = float(current_cycle)
        s0 = _survival_at_step(series, t0)
        end = int(t0) + max_future
        cycles = list(range(int(t0), end + 1))
        probs = [
            _survival_at_step(series, float(t)) / s0 if s0 > 1e-9 else 0.0 for t in cycles
        ]
        return pd.DataFrame({"cycle": cycles, "survival_prob": np.clip(probs, 0, 1)})

    def top_hazard_ratios(self, n: int = 8) -> pd.DataFrame:
        """Interpretability: features most associated with higher hazard."""
        self._ensure_fitted()
        summary = self.model.summary[["coef", "exp(coef)", "p"]].copy()
        summary = summary.rename(
            columns={"coef": "coef", "exp(coef)": "hazard_ratio", "p": "p_value"}
        )
        summary["abs_coef"] = summary["coef"].abs()
        return summary.sort_values("abs_coef", ascending=False).head(n)

    def _ensure_fitted(self) -> None:
        if not self._fitted or self.model is None:
            raise RuntimeError("SurvivalModel is not fitted")

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "penalizer": self.penalizer,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SurvivalModel":
        import joblib

        data = joblib.load(path)
        instance = cls(penalizer=data.get("penalizer", 0.05))
        instance.model = data["model"]
        instance.feature_cols = data["feature_cols"]
        instance._fitted = data.get("fitted", True)
        return instance
