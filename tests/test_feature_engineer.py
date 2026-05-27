"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_cmapss():
    np.random.seed(0)
    rows = []
    for unit in [1, 2]:
        for cycle in range(1, 21):
            row = {"unit_id": unit, "cycle": cycle}
            for i in range(1, 22):
                row[f"sensor_{i}"] = np.random.randn()
            rows.append(row)
    return pd.DataFrame(rows)


def test_rolling_features_add_columns(sample_cmapss):
    engineer = FeatureEngineer(window_sizes=[5])
    result = engineer.add_rolling_features(sample_cmapss)
    assert "sensor_1_roll5_mean" in result.columns
    assert "sensor_1_roll5_std" in result.columns


def test_lag_features(sample_cmapss):
    engineer = FeatureEngineer()
    result = engineer.add_lag_features(sample_cmapss, lags=[1])
    assert "sensor_1_lag1" in result.columns
    assert pd.isna(result.loc[0, "sensor_1_lag1"])


def test_degradation_index(sample_cmapss):
    engineer = FeatureEngineer()
    result = engineer.add_degradation_index(sample_cmapss)
    assert "degradation_index" in result.columns
    assert result["degradation_index"].min() >= 0


def test_engineer_cmapss_pipeline(sample_cmapss):
    engineer = FeatureEngineer(window_sizes=[3])
    result = engineer.engineer_cmapss(sample_cmapss)
    assert "degradation_index" in result.columns
    assert len(result) == len(sample_cmapss)
