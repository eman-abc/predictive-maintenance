"""Tests for CMAPSS Phase 2 labeling, preprocessing, and pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion.cmapss_loader import (
    add_failure_horizon_labels,
    compute_test_rul,
    compute_train_rul,
)
from src.ingestion.cmapss_preprocessor import CmapssPreprocessor
from src.ingestion.cmapss_pipeline import build_cmapss_dataset

RAW_DIR = Path("data/raw/cmapss")
HAS_FD001 = (RAW_DIR / "train_FD001.txt").exists()


def _synthetic_unit(unit_id: int, n_cycles: int, base: float = 100.0) -> pd.DataFrame:
    rows = []
    for cycle in range(1, n_cycles + 1):
        row = {
            "unit_id": unit_id,
            "cycle": cycle,
            "op_setting_1": 0.0,
            "op_setting_2": 0.0,
            "op_setting_3": 100.0,
        }
        for i in range(1, 22):
            row[f"sensor_{i}"] = base + cycle * 0.1 + i
        rows.append(row)
    return pd.DataFrame(rows)


def test_compute_train_rul_caps_at_max():
    df = _synthetic_unit(1, 200)
    out = compute_train_rul(df, cap=125)
    assert out["rul"].max() == 125
    assert out.loc[out["cycle"] == 200, "rul"].iloc[0] == 0


def test_compute_test_rul_matches_label_at_last_cycle():
    df = pd.concat([_synthetic_unit(1, 50), _synthetic_unit(2, 40)])
    rul_end = pd.Series([30, 10])
    out = compute_test_rul(df, rul_end, cap=None)
    last1 = out[(out["unit_id"] == 1) & (out["cycle"] == 50)]["rul"].iloc[0]
    last2 = out[(out["unit_id"] == 2) & (out["cycle"] == 40)]["rul"].iloc[0]
    assert last1 == 30
    assert last2 == 10
    mid = out[(out["unit_id"] == 1) & (out["cycle"] == 45)]["rul"].iloc[0]
    assert mid == 35


def test_failure_horizon_labels():
    df = pd.DataFrame({"rul": [10, 30, 72, 100]})
    out = add_failure_horizon_labels(df, horizons=[30])
    assert out["failure_30"].tolist() == [1, 1, 0, 0]


def test_preprocessor_per_unit_no_cross_leakage():
    train = pd.concat([_synthetic_unit(1, 20, base=100), _synthetic_unit(2, 20, base=500)])
    cfg_sensors = [f"sensor_{i}" for i in range(2, 22)]
    prep = CmapssPreprocessor(sensor_cols=cfg_sensors[:3], drop_sensors=["sensor_1"])
    prep.fit(train)
    out = prep.transform(train, is_train=True)
    for col in prep.sensor_cols:
        early = out[out["unit_id"] == 1].nsmallest(5, "cycle")[col]
        assert abs(early.mean()) < 0.5


@pytest.mark.skipif(not HAS_FD001, reason="CMAPSS raw data not present")
def test_build_fd001_dataset_writes_parquet(tmp_path):
    result = build_cmapss_dataset(
        "FD001",
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )
    train = pd.read_parquet(result["train_path"])
    test = pd.read_parquet(result["test_path"])
    assert "rul" in train.columns
    assert "failure_30" in train.columns
    assert train["rul"].max() < 125
    assert len(test) > 0
    assert len(result["meta"]["feature_columns"]) > 50
