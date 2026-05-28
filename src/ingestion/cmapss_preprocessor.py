"""Leakage-safe preprocessing for CMAPSS (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]


@dataclass
class CmapssPreprocessor:
    """
    Fit on training data only; transform train and test.

  Steps:
  1. Drop low-value sensors (from Phase 1 config).
  2. Assign operating-condition clusters (FD002/004) — KMeans fit on train.
  3. Per-unit sensor baseline normalization (early cycles per engine).
  4. Optional cluster-wise StandardScaler (fit on train only).
  """

    sensor_cols: list[str]
    drop_sensors: list[str] = field(default_factory=list)
    baseline_cycles: int = 5
    cluster_for_normalization: bool = False
    n_op_clusters: int = 6
    op_kmeans: KMeans | None = None
    cluster_scalers: dict[int, StandardScaler] = field(default_factory=dict)
    unit_baseline_mean_: dict[int, pd.Series] = field(default_factory=dict)
    unit_baseline_std_: dict[int, pd.Series] = field(default_factory=dict)

    def _drop_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.drop_sensors if c in df.columns]
        return df.drop(columns=cols, errors="ignore")

    def fit_op_clusters(self, train: pd.DataFrame) -> None:
        if not self.cluster_for_normalization:
            return
        self.op_kmeans = KMeans(
            n_clusters=self.n_op_clusters,
            random_state=42,
            n_init=10,
        )
        self.op_kmeans.fit(train[OP_COLS].round(2))

    def transform_op_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.op_kmeans is None:
            df["op_cluster"] = 0
            return df
        df["op_cluster"] = self.op_kmeans.predict(df[OP_COLS].round(2))
        return df

    @staticmethod
    def _baseline_stats(
        group: pd.DataFrame, sensor_cols: list[str], baseline_cycles: int
    ) -> tuple[pd.Series, pd.Series]:
        early = group.nsmallest(baseline_cycles, "cycle")
        mean = early[sensor_cols].mean()
        std = early[sensor_cols].std().replace(0, np.nan).fillna(1.0)
        return mean, std

    def _fit_unit_baselines(self, train: pd.DataFrame) -> None:
        self.unit_baseline_mean_.clear()
        self.unit_baseline_std_.clear()
        for unit_id, group in train.groupby("unit_id"):
            mean, std = self._baseline_stats(group, self.sensor_cols, self.baseline_cycles)
            self.unit_baseline_mean_[unit_id] = mean
            self.unit_baseline_std_[unit_id] = std

    def _apply_unit_baselines(
        self, df: pd.DataFrame, *, use_train_baselines: bool
    ) -> pd.DataFrame:
        df = df.copy()
        for unit_id, group in df.groupby("unit_id"):
            if use_train_baselines and unit_id in self.unit_baseline_mean_:
                mean = self.unit_baseline_mean_[unit_id]
                std = self.unit_baseline_std_[unit_id].fillna(1.0).replace(0, 1.0)
            else:
                mean, std = self._baseline_stats(
                    group, self.sensor_cols, self.baseline_cycles
                )
            idx = group.index
            df.loc[idx, self.sensor_cols] = (
                group[self.sensor_cols] - mean
            ) / (std + 1e-8)
        return df

    def fit_cluster_scalers(self, train: pd.DataFrame) -> None:
        if not self.cluster_for_normalization:
            return
        self.cluster_scalers.clear()
        for cluster_id in sorted(train["op_cluster"].unique()):
            mask = train["op_cluster"] == cluster_id
            scaler = StandardScaler()
            scaler.fit(train.loc[mask, self.sensor_cols])
            self.cluster_scalers[int(cluster_id)] = scaler

    def _apply_cluster_scalers(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cluster_scalers:
            return df
        df = df.copy()
        out = df[self.sensor_cols].astype(float).copy()
        for cluster_id, scaler in self.cluster_scalers.items():
            mask = df["op_cluster"] == cluster_id
            if mask.any():
                out.loc[mask] = scaler.transform(df.loc[mask, self.sensor_cols])
        for col in self.sensor_cols:
            df[col] = out[col]
        return df

    def fit(self, train: pd.DataFrame) -> CmapssPreprocessor:
        train = self._drop_sensors(train)
        self.fit_op_clusters(train)
        train = self.transform_op_clusters(train)
        self._fit_unit_baselines(train)
        train_norm = self._apply_unit_baselines(train, use_train_baselines=True)
        self.fit_cluster_scalers(train_norm)
        return self

    def transform(self, df: pd.DataFrame, *, is_train: bool = False) -> pd.DataFrame:
        df = self._drop_sensors(df)
        df = self.transform_op_clusters(df)
        df = self._apply_unit_baselines(df, use_train_baselines=is_train)
        return self._apply_cluster_scalers(df)

    @classmethod
    def from_config(cls, config: dict) -> CmapssPreprocessor:
        sensors = config["sensors"]
        op_cfg = config["operating_conditions"]
        return cls(
            sensor_cols=sensors["keep"],
            drop_sensors=sensors.get("drop", []),
            cluster_for_normalization=op_cfg.get("cluster_for_normalization", False),
            n_op_clusters=op_cfg.get("n_scenarios", 6),
        )
