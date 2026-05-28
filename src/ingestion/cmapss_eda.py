"""Exploratory data analysis utilities for NASA CMAPSS datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.cmapss_loader import (
    CMAPSS_COLUMNS,
    load_cmapss_rul,
    load_cmapss_test,
    load_cmapss_train,
)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]

DATASET_META: dict[str, dict[str, Any]] = {
    "FD001": {"n_operating_conditions": 1, "n_fault_modes": 1},
    "FD002": {"n_operating_conditions": 6, "n_fault_modes": 1},
    "FD003": {"n_operating_conditions": 1, "n_fault_modes": 2},
    "FD004": {"n_operating_conditions": 6, "n_fault_modes": 2},
}

# Literature-reported near-constant sensors for FD001 (used as prior; EDA confirms per dataset).
FD001_LITERATURE_DROP = {
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
}


@dataclass
class SensorVarianceReport:
    """Per-sensor variance statistics for one split."""

    dataset_id: str
    split: str
    constant_sensors: list[str] = field(default_factory=list)
    near_constant_sensors: list[str] = field(default_factory=list)
    informative_sensors: list[str] = field(default_factory=list)
    std_by_sensor: dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetEdaReport:
    """Aggregated Phase-1 EDA results for one FD subset."""

    dataset_id: str
    meta: dict[str, Any]
    train_units: int
    test_units: int
    train_rows: int
    test_rows: int
    train_cycle_min: int
    train_cycle_max: int
    train_cycle_median: float
    test_cycle_min: int
    test_cycle_max: int
    test_cycle_median: float
    train_run_length: dict[str, float]
    test_run_length: dict[str, float]
    implied_remaining_from_rul: dict[str, float]
    op_setting_nunique: dict[str, int]
    n_unique_op_triplets: int
    per_unit_op_stable: bool
    cluster_for_normalization: bool
    train_sensor_report: SensorVarianceReport
    test_sensor_report: SensorVarianceReport
    recommended_drop_sensors: list[str]
    recommended_keep_sensors: list[str]
    rul_at_last_cycle: dict[str, float]


def _sensor_variance_report(
    df: pd.DataFrame, dataset_id: str, split: str, std_threshold: float = 1e-6
) -> SensorVarianceReport:
    stds = df[SENSOR_COLS].std()
    constant = [c for c in SENSOR_COLS if stds[c] == 0 or np.isnan(stds[c])]
    near_constant = [
        c
        for c in SENSOR_COLS
        if c not in constant and stds[c] < std_threshold
    ]
    informative = [
        c for c in SENSOR_COLS if c not in constant and c not in near_constant
    ]
    return SensorVarianceReport(
        dataset_id=dataset_id,
        split=split,
        constant_sensors=constant,
        near_constant_sensors=near_constant,
        informative_sensors=informative,
        std_by_sensor={c: float(stds[c]) for c in SENSOR_COLS},
    )


def _run_length_stats(cycles_per_unit: pd.Series) -> dict[str, float]:
    return {
        "min": float(cycles_per_unit.min()),
        "max": float(cycles_per_unit.max()),
        "mean": float(cycles_per_unit.mean()),
        "median": float(cycles_per_unit.median()),
        "std": float(cycles_per_unit.std()),
    }


def _implied_failure_cycle(test: pd.DataFrame, rul: pd.Series) -> dict[str, float]:
    """
    For each test engine, NASA RUL is remaining cycles after the last observed row.
    Implied failure cycle = last_observed_cycle + RUL (per engine, ordered by unit_id).
    """
    test_max = test.groupby("unit_id")["cycle"].max().sort_index()
    if len(test_max) != len(rul):
        rul = rul.reset_index(drop=True)
    implied = test_max.values + rul.values[: len(test_max)]
    return {
        "min": float(np.min(implied)),
        "max": float(np.max(implied)),
        "mean": float(np.mean(implied)),
        "median": float(np.median(implied)),
    }


def _op_condition_stats(train: pd.DataFrame) -> tuple[dict[str, int], int, bool]:
    """
    Operating settings vary across engines; per-column nunique is high even for FD001.

    We report:
    - per-column nunique (diagnostic),
    - unique (op1, op2, op3) triplets rounded to 2 decimals,
    - whether each unit holds stable settings across its life (std < 1e-4).
    """
    op_nunique = {c: int(train[c].nunique()) for c in OP_COLS}
    rounded = train[OP_COLS].round(2)
    n_triplets = int(rounded.drop_duplicates().shape[0])
    per_unit_std = train.groupby("unit_id")[OP_COLS].std()
    stable = bool((per_unit_std < 1e-4).all().all())
    return op_nunique, n_triplets, stable


def _recommend_sensor_drops(
    train_report: SensorVarianceReport,
    test_report: SensorVarianceReport,
    dataset_id: str,
) -> tuple[list[str], list[str]]:
    """Union constant/near-constant across train+test; cross-check FD001 literature prior."""
    drop = set(train_report.constant_sensors) | set(test_report.constant_sensors)
    drop |= set(train_report.near_constant_sensors) | set(test_report.near_constant_sensors)

    if dataset_id == "FD001":
        drop |= FD001_LITERATURE_DROP

    drop_sorted = sorted(drop, key=lambda s: int(s.split("_")[1]))
    keep = [c for c in SENSOR_COLS if c not in drop]
    return drop_sorted, keep


def analyze_dataset(
    data_dir: str | Path,
    dataset_id: str = "FD001",
    std_threshold: float = 1e-6,
) -> DatasetEdaReport:
    """Run full Phase-1 EDA for one CMAPSS subset (FD001–FD004)."""
    data_dir = Path(data_dir)
    train = load_cmapss_train(data_dir, dataset_id)
    test = load_cmapss_test(data_dir, dataset_id)
    rul = load_cmapss_rul(data_dir, dataset_id)

    train_cycles = train.groupby("unit_id")["cycle"].max()
    test_cycles = test.groupby("unit_id")["cycle"].max()

    train_report = _sensor_variance_report(train, dataset_id, "train", std_threshold)
    test_report = _sensor_variance_report(test, dataset_id, "test", std_threshold)
    drop, keep = _recommend_sensor_drops(train_report, test_report, dataset_id)

    meta = DATASET_META[dataset_id]
    op_nunique, n_triplets, per_unit_stable = _op_condition_stats(train)
    cluster_norm = meta["n_operating_conditions"] > 1

    return DatasetEdaReport(
        dataset_id=dataset_id,
        meta=meta,
        train_units=int(train["unit_id"].nunique()),
        test_units=int(test["unit_id"].nunique()),
        train_rows=len(train),
        test_rows=len(test),
        train_cycle_min=int(train_cycles.min()),
        train_cycle_max=int(train_cycles.max()),
        train_cycle_median=float(train_cycles.median()),
        test_cycle_min=int(test_cycles.min()),
        test_cycle_max=int(test_cycles.max()),
        test_cycle_median=float(test_cycles.median()),
        train_run_length=_run_length_stats(train_cycles),
        test_run_length=_run_length_stats(test_cycles),
        implied_remaining_from_rul=_implied_failure_cycle(test, rul),
        op_setting_nunique=op_nunique,
        n_unique_op_triplets=n_triplets,
        per_unit_op_stable=per_unit_stable,
        cluster_for_normalization=cluster_norm,
        train_sensor_report=train_report,
        test_sensor_report=test_report,
        recommended_drop_sensors=drop,
        recommended_keep_sensors=keep,
        rul_at_last_cycle={
            "min": float(rul.min()),
            "max": float(rul.max()),
            "mean": float(rul.mean()),
            "median": float(rul.median()),
        },
    )


def analyze_all(
    data_dir: str | Path,
    datasets: list[str] | None = None,
) -> dict[str, DatasetEdaReport]:
    """Analyze all requested FD subsets."""
    datasets = datasets or ["FD001", "FD002", "FD003", "FD004"]
    return {ds: analyze_dataset(data_dir, ds) for ds in datasets}


def report_to_config_dict(report: DatasetEdaReport, rul_cap: int = 125) -> dict[str, Any]:
    """Serialize EDA decisions into a pipeline config dictionary."""
    return {
        "dataset_id": report.dataset_id,
        "description": (
            f"CMAPSS {report.dataset_id}: "
            f"{report.meta['n_operating_conditions']} operating condition(s), "
            f"{report.meta['n_fault_modes']} fault mode(s)"
        ),
        "paths": {
            "raw_dir": "./data/raw/cmapss",
            "train_file": f"train_{report.dataset_id}.txt",
            "test_file": f"test_{report.dataset_id}.txt",
            "rul_file": f"RUL_{report.dataset_id}.txt",
        },
        "labels": {
            "rul_cap": rul_cap,
            "failure_horizons_cycles": [30, 72],
        },
        "sensors": {
            "drop": report.recommended_drop_sensors,
            "keep": report.recommended_keep_sensors,
        },
        "operating_conditions": {
            "n_scenarios": report.meta["n_operating_conditions"],
            "nunique_per_column": report.op_setting_nunique,
            "n_unique_op_triplets": report.n_unique_op_triplets,
            "per_unit_op_stable": report.per_unit_op_stable,
            "cluster_for_normalization": report.cluster_for_normalization,
        },
        "feature_engineering": {
            "rolling_windows": [5, 10, 30],
            "lag_cycles": [1, 3, 5],
            "include_rate_of_change": True,
            "include_rolling_slope": True,
            "include_spectral": True,
            "spectral_sensors": report.recommended_keep_sensors[:5],
            "train_row_filter_max_rul": rul_cap,
        },
        "eda_snapshot": {
            "train_units": report.train_units,
            "test_units": report.test_units,
            "train_cycle_range": [report.train_cycle_min, report.train_cycle_max],
            "test_cycle_range": [report.test_cycle_min, report.test_cycle_max],
            "train_run_length": report.train_run_length,
            "test_run_length": report.test_run_length,
            "implied_failure_cycle": report.implied_remaining_from_rul,
            "rul_at_last_cycle": report.rul_at_last_cycle,
        },
    }
