"""Tests for config-driven CmapssFeatureEngineer (Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.cmapss_config import load_cmapss_config
from src.ingestion.feature_engineer import CmapssFeatureEngineer


def _synthetic_preprocessed(
    *,
    units: tuple[int, ...] = (1, 2),
    n_cycles: int = 40,
    sensors: tuple[str, ...] = ("sensor_2", "sensor_3", "sensor_4"),
) -> pd.DataFrame:
    """Minimal post-preprocessor frame (kept sensors only)."""
    rows = []
    for unit in units:
        for cycle in range(1, n_cycles + 1):
            row: dict = {
                "unit_id": unit,
                "cycle": cycle,
                "rul": max(0, n_cycles - cycle),
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
                "op_cluster": 0,
            }
            for i, s in enumerate(sensors):
                row[s] = float(unit * 10 + cycle * 0.5 + i)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def minimal_fe_config() -> dict:
    return {
        "sensors": {"keep": ["sensor_2", "sensor_3", "sensor_4"]},
        "feature_engineering": {
            "rolling_windows": [5],
            "lag_cycles": [1],
            "include_rate_of_change": True,
            "include_rolling_slope": False,
            "include_spectral": False,
            "spectral_sensors": ["sensor_2"],
            "spectral_window": 10,
        },
    }


def test_from_config_builds_engineer(minimal_fe_config):
    eng = CmapssFeatureEngineer.from_config(minimal_fe_config)
    assert eng.sensor_cols == ["sensor_2", "sensor_3", "sensor_4"]
    assert eng.window_sizes == [5]
    assert eng.include_rolling_slope is False


def test_transform_adds_rolling_lag_delta(minimal_fe_config):
    df = _synthetic_preprocessed()
    eng = CmapssFeatureEngineer.from_config(minimal_fe_config)
    out = eng.transform(df)
    assert "sensor_2_roll5_mean" in out.columns
    assert "sensor_2_lag1" in out.columns
    assert "sensor_2_delta1" in out.columns
    assert "degradation_index" in out.columns
    assert len(out) == len(df)


def test_lag_first_cycle_is_nan_per_unit(minimal_fe_config):
    df = _synthetic_preprocessed(units=(1, 2), n_cycles=10)
    eng = CmapssFeatureEngineer.from_config(minimal_fe_config)
    out = eng.transform(df)
    first = out[(out["unit_id"] == 1) & (out["cycle"] == 1)]["sensor_2_lag1"].iloc[0]
    assert pd.isna(first)


def test_spectral_features_when_enabled(minimal_fe_config):
    minimal_fe_config["feature_engineering"]["include_spectral"] = True
    df = _synthetic_preprocessed(n_cycles=30)
    eng = CmapssFeatureEngineer.from_config(minimal_fe_config)
    out = eng.transform(df)
    assert "sensor_2_spectral10_power" in out.columns


def test_feature_column_names_excludes_raw_sensors_and_labels(minimal_fe_config):
    df = _synthetic_preprocessed()
    eng = CmapssFeatureEngineer.from_config(minimal_fe_config)
    out = eng.transform(df)
    cols = CmapssFeatureEngineer.feature_column_names(out)
    assert "sensor_2" not in cols
    assert "rul" not in cols
    assert "unit_id" not in cols
    assert "degradation_index" in cols
    assert all(c in out.columns for c in cols)


def test_fd001_config_engineer_produces_many_features():
    cfg = load_cmapss_config("FD001")
    df = _synthetic_preprocessed(
        sensors=tuple(cfg["sensors"]["keep"]),
        n_cycles=35,
    )
    eng = CmapssFeatureEngineer.from_config(cfg)
    out = eng.transform(df)
    model_cols = CmapssFeatureEngineer.feature_column_names(out)
    assert len(model_cols) >= 20
    assert any("spectral" in c for c in model_cols)
    assert any("slope" in c for c in model_cols)
