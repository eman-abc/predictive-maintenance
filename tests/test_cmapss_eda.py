"""Tests for CMAPSS Phase-1 EDA utilities."""

from pathlib import Path

import pytest

from src.ingestion.cmapss_eda import (
    FD001_LITERATURE_DROP,
    analyze_dataset,
    report_to_config_dict,
)

RAW_DIR = Path("data/raw/cmapss")
SKIP_NO_DATA = not (RAW_DIR / "train_FD001.txt").exists()


@pytest.mark.skipif(SKIP_NO_DATA, reason="CMAPSS raw data not present")
def test_fd001_eda_counts_match_readme():
    report = analyze_dataset(RAW_DIR, "FD001")
    assert report.train_units == 100
    assert report.test_units == 100
    assert report.meta["n_fault_modes"] == 1


@pytest.mark.skipif(SKIP_NO_DATA, reason="CMAPSS raw data not present")
def test_fd001_drops_include_literature_sensors():
    report = analyze_dataset(RAW_DIR, "FD001")
    for sensor in FD001_LITERATURE_DROP:
        assert sensor in report.recommended_drop_sensors


@pytest.mark.skipif(SKIP_NO_DATA, reason="CMAPSS raw data not present")
def test_fd002_requires_op_clustering():
    report = analyze_dataset(RAW_DIR, "FD002")
    assert report.cluster_for_normalization is True


@pytest.mark.skipif(SKIP_NO_DATA, reason="CMAPSS raw data not present")
def test_fd001_no_op_clustering():
    report = analyze_dataset(RAW_DIR, "FD001")
    assert report.cluster_for_normalization is False


def test_config_dict_has_required_keys():
    from src.ingestion.cmapss_eda import DatasetEdaReport, SensorVarianceReport

    stub = DatasetEdaReport(
        dataset_id="FD001",
        meta={"n_operating_conditions": 1, "n_fault_modes": 1},
        train_units=100,
        test_units=100,
        train_rows=1000,
        test_rows=500,
        train_cycle_min=1,
        train_cycle_max=200,
        train_cycle_median=150.0,
        test_cycle_min=1,
        test_cycle_max=150,
        test_cycle_median=80.0,
        train_run_length={"min": 1, "max": 200, "mean": 150, "median": 150, "std": 10},
        test_run_length={"min": 1, "max": 150, "mean": 80, "median": 80, "std": 10},
        implied_remaining_from_rul={"min": 100, "max": 250, "mean": 180, "median": 175},
        op_setting_nunique={"op_setting_1": 1, "op_setting_2": 1, "op_setting_3": 1},
        n_unique_op_triplets=1,
        per_unit_op_stable=True,
        cluster_for_normalization=False,
        train_sensor_report=SensorVarianceReport("FD001", "train", informative_sensors=["sensor_2"]),
        test_sensor_report=SensorVarianceReport("FD001", "test", informative_sensors=["sensor_2"]),
        recommended_drop_sensors=["sensor_1"],
        recommended_keep_sensors=["sensor_2"],
        rul_at_last_cycle={"min": 7, "max": 145, "mean": 50, "median": 45},
    )
    cfg = report_to_config_dict(stub)
    assert cfg["labels"]["rul_cap"] == 125
    assert "sensor_2" in cfg["sensors"]["keep"]
