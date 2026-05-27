"""Feature engineering for CMAPSS and AI4I datasets."""

from pathlib import Path

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Rolling-window and lag features for predictive maintenance."""

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
        """Full feature pipeline for CMAPSS data."""
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
