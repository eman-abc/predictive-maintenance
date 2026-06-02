"""Feature engineering for CMAPSS and AI4I datasets."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LEGACY_CMAPSS_FE_MSG = (
    "FeatureEngineer.engineer_cmapss() is deprecated. Use Phase 2 instead: "
    "build_cmapss_dataset() with CmapssPreprocessor + CmapssFeatureEngineer "
    "(see scripts/build_cmapss_dataset.py and docs/cmapss_phase2_preprocessing.md)."
)

LABEL_COLS = {
    "unit_id",
    "cycle",
    "rul",
    "op_cluster",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
}
LABEL_COLS |= {f"failure_{h}" for h in (30, 72)}


class FeatureEngineer:
    """
    Legacy helpers for rolling/lag features and AI4I.

    CMAPSS: use :class:`CmapssFeatureEngineer` via ``build_cmapss_dataset()`` — not this class.
    """

    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
    OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]

    def __init__(self, window_sizes: list[int] | None = None):
        self.window_sizes = window_sizes or [5, 10, 30]

    def add_rolling_features(
        self, df: pd.DataFrame, group_col: str = "unit_id"
    ) -> pd.DataFrame:
        """Add rolling mean, std, and trend for sensor columns."""
        df = df.sort_values([group_col, "cycle"]).copy()
        features = []

        for window in self.window_sizes:
            grouped = df.groupby(group_col)[self.SENSOR_COLS]
            roll_mean = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            roll_std = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
            roll_mean.columns = [f"{c}_roll{window}_mean" for c in self.SENSOR_COLS]
            roll_std.columns = [f"{c}_roll{window}_std" for c in self.SENSOR_COLS]
            features.extend([roll_mean, roll_std])

        return pd.concat([df] + features, axis=1)

    def add_lag_features(
        self, df: pd.DataFrame, lags: list[int] | None = None, group_col: str = "unit_id"
    ) -> pd.DataFrame:
        """Add lagged sensor values."""
        df = df.sort_values([group_col, "cycle"]).copy()
        lags = lags or [1, 3, 5]
        lag_frames = []

        for lag in lags:
            lagged = df.groupby(group_col)[self.SENSOR_COLS].shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in self.SENSOR_COLS]
            lag_frames.append(lagged)

        return pd.concat([df] + lag_frames, axis=1)

    def add_degradation_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a simple degradation index from normalized sensor drift."""
        df = df.copy()
        sensor_data = df[self.SENSOR_COLS]
        normalized = (sensor_data - sensor_data.mean()) / (sensor_data.std() + 1e-8)
        df["degradation_index"] = normalized.abs().mean(axis=1)
        return df

    def engineer_cmapss(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deprecated CMAPSS feature pipeline (all 21 sensors, no Phase 2 preprocessing).

        Use ``src.ingestion.cmapss_pipeline.build_cmapss_dataset`` instead.
        """
        warnings.warn(_LEGACY_CMAPSS_FE_MSG, DeprecationWarning, stacklevel=2)
        df = self.add_rolling_features(df)
        df = self.add_lag_features(df)
        df = self.add_degradation_index(df)
        return df

    def engineer_ai4i(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature pipeline for AI4I tabular data."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {"UDI", "Product ID", "failure", "Machine failure"}
        feature_cols = [c for c in numeric_cols if c not in exclude]

        if feature_cols:
            df["feature_mean"] = df[feature_cols].mean(axis=1)
            df["feature_std"] = df[feature_cols].std(axis=1).fillna(0)
        return df

    def save_processed(self, df: pd.DataFrame, output_path: str | Path) -> Path:
        """Persist engineered features as Parquet."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path


@dataclass
class CmapssFeatureEngineer:
    """
    Canonical CMAPSS feature pipeline (Phase 2).

    Config-driven rolling, lag, delta, slope, spectral, and degradation_index features.
    Invoked from ``build_cmapss_dataset()`` after ``CmapssPreprocessor``.
    """

    sensor_cols: list[str]
    window_sizes: list[int] = field(default_factory=lambda: [5, 10, 30])
    lag_cycles: list[int] = field(default_factory=lambda: [1, 3, 5])
    include_rate_of_change: bool = True
    include_rolling_slope: bool = True
    include_spectral: bool = True
    spectral_sensors: list[str] = field(default_factory=list)
    spectral_window: int = 10

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CmapssFeatureEngineer:
        fe = config["feature_engineering"]
        sensors = config["sensors"]["keep"]
        return cls(
            sensor_cols=sensors,
            window_sizes=fe.get("rolling_windows", [5, 10, 30]),
            lag_cycles=fe.get("lag_cycles", [1, 3, 5]),
            include_rate_of_change=fe.get("include_rate_of_change", True),
            include_rolling_slope=fe.get("include_rolling_slope", True),
            include_spectral=fe.get("include_spectral", True),
            spectral_sensors=fe.get("spectral_sensors", sensors[:5]),
            spectral_window=fe.get("spectral_window", 10),
        )

    def add_rolling_features(
        self, df: pd.DataFrame, group_col: str = "unit_id"
    ) -> pd.DataFrame:
        df = df.sort_values([group_col, "cycle"]).copy()
        parts = [df]
        for window in self.window_sizes:
            grouped = df.groupby(group_col)[self.sensor_cols]
            roll_mean = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            roll_std = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
            roll_mean.columns = [f"{c}_roll{window}_mean" for c in self.sensor_cols]
            roll_std.columns = [f"{c}_roll{window}_std" for c in self.sensor_cols]
            parts.extend([roll_mean, roll_std])
        return pd.concat(parts, axis=1)

    def add_lag_features(
        self, df: pd.DataFrame, group_col: str = "unit_id"
    ) -> pd.DataFrame:
        df = df.sort_values([group_col, "cycle"]).copy()
        parts = [df]
        for lag in self.lag_cycles:
            lagged = df.groupby(group_col)[self.sensor_cols].shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in self.sensor_cols]
            parts.append(lagged)
        return pd.concat(parts, axis=1)

    def add_rate_of_change(
        self, df: pd.DataFrame, group_col: str = "unit_id"
    ) -> pd.DataFrame:
        df = df.sort_values([group_col, "cycle"]).copy()
        delta = df.groupby(group_col)[self.sensor_cols].diff()
        delta.columns = [f"{c}_delta1" for c in self.sensor_cols]
        return pd.concat([df, delta], axis=1)

    def add_rolling_slope(
        self, df: pd.DataFrame, group_col: str = "unit_id"
    ) -> pd.DataFrame:
        df = df.sort_values([group_col, "cycle"]).copy()
        parts = [df]
        for window in self.window_sizes:
            grouped = df.groupby(group_col)[self.sensor_cols]

            def _slope(arr: np.ndarray) -> float:
                if len(arr) < 2:
                    return 0.0
                x = np.arange(len(arr))
                return float(np.polyfit(x, np.asarray(arr, dtype=float), 1)[0])

            slope = grouped.transform(
                lambda col: col.rolling(window, min_periods=2).apply(_slope, raw=True)
            )
            slope.columns = [f"{c}_roll{window}_slope" for c in self.sensor_cols]
            parts.append(slope.fillna(0))
        return pd.concat(parts, axis=1)

    def add_spectral_features(
        self, df: pd.DataFrame, group_col: str = "unit_id"
    ) -> pd.DataFrame:
        df = df.sort_values([group_col, "cycle"]).copy()
        parts = [df]
        w = self.spectral_window
        for sensor in self.spectral_sensors:
            if sensor not in df.columns:
                continue

            def _band_power(arr: np.ndarray) -> float:
                if len(arr) < 4:
                    return 0.0
                y = np.asarray(arr, dtype=float)
                y = y - y.mean()
                spectrum = np.abs(np.fft.rfft(y))
                return float(np.sum(spectrum**2) / len(spectrum))

            power = (
                df.groupby(group_col)[sensor]
                .transform(lambda s: s.rolling(w, min_periods=4).apply(_band_power, raw=True))
                .fillna(0)
            )
            parts.append(power.rename(f"{sensor}_spectral{w}_power"))
        return pd.concat(parts, axis=1)

    def add_degradation_index(
        self, df: pd.DataFrame, group_col: str = "unit_id"
    ) -> pd.DataFrame:
        """Per-unit normalized mean absolute sensor level (health proxy)."""
        df = df.copy()

        def _degrad(group: pd.DataFrame) -> pd.Series:
            block = group[self.sensor_cols]
            norm = (block - block.mean()) / (block.std() + 1e-8)
            return norm.abs().mean(axis=1)

        df["degradation_index"] = df.groupby(group_col, group_keys=False).apply(
            _degrad, include_groups=False
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.add_rolling_features(df)
        out = self.add_lag_features(out)
        if self.include_rate_of_change:
            out = self.add_rate_of_change(out)
        if self.include_rolling_slope:
            out = self.add_rolling_slope(out)
        if self.include_spectral:
            out = self.add_spectral_features(out)
        return self.add_degradation_index(out)

    @staticmethod
    def feature_column_names(df: pd.DataFrame) -> list[str]:
        raw_sensors = {f"sensor_{i}" for i in range(1, 22)}
        skip = LABEL_COLS | raw_sensors | {
            "op_setting_1",
            "op_setting_2",
            "op_setting_3",
        }
        return [
            c
            for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])
        ]
