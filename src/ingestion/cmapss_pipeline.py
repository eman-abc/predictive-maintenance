"""End-to-end CMAPSS Phase 2 dataset builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.ingestion.cmapss_config import load_cmapss_config
from src.ingestion.cmapss_loader import (
    add_failure_horizon_labels,
    compute_test_rul,
    compute_train_rul,
    load_cmapss_rul,
    load_cmapss_test,
    load_cmapss_train,
)
from src.ingestion.cmapss_preprocessor import CmapssPreprocessor
from src.ingestion.feature_engineer import CmapssFeatureEngineer


def _output_paths(
    processed_dir: Path, artifacts_dir: Path, dataset_id: str
) -> tuple[Path, Path, Path, Path]:
    return (
        processed_dir / f"cmapss_{dataset_id}_train.parquet",
        processed_dir / f"cmapss_{dataset_id}_test.parquet",
        artifacts_dir / f"cmapss_{dataset_id}_preprocessor.joblib",
        artifacts_dir / f"cmapss_{dataset_id}_feature_columns.json",
    )


def build_cmapss_dataset(
    dataset_id: str = "FD001",
    config_dir: str | Path = "configs",
    processed_dir: str | Path = "data/processed",
    artifacts_dir: str | Path = "artifacts",
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """
    Phase 2 pipeline: labels → preprocess → features → persist.

    Returns paths and row counts; writes Parquet + preprocessor artifact when
    `write_outputs` is True.
    """
    config = load_cmapss_config(dataset_id, config_dir)
    raw_dir = Path(config["paths"]["raw_dir"])
    labels_cfg = config["labels"]
    fe_cfg = config["feature_engineering"]
    cap = labels_cfg["rul_cap"]
    horizons = labels_cfg.get("failure_horizons_cycles", [30, 72])

    train = load_cmapss_train(raw_dir, dataset_id)
    test = load_cmapss_test(raw_dir, dataset_id)
    rul_end = load_cmapss_rul(raw_dir, dataset_id)

    train = compute_train_rul(train, cap=cap)
    test = compute_test_rul(test, rul_end, cap=cap)
    train = add_failure_horizon_labels(train, horizons)
    test = add_failure_horizon_labels(test, horizons)

    preprocessor = CmapssPreprocessor.from_config(config)
    preprocessor.fit(train)
    train = preprocessor.transform(train, is_train=True)
    test = preprocessor.transform(test, is_train=False)

    engineer = CmapssFeatureEngineer.from_config(config)
    train = engineer.transform(train)
    test = engineer.transform(test)

    max_rul = fe_cfg.get("train_row_filter_max_rul", cap)
    # Exclude capped healthy plateau (rul == cap); keep degrading region only.
    train_filtered = train[train["rul"] < max_rul].copy()

    meta = {
        "dataset_id": dataset_id,
        "train_rows_raw": len(train),
        "train_rows_filtered": len(train_filtered),
        "test_rows": len(test),
        "feature_columns": engineer.feature_column_names(train_filtered),
    }

    if not write_outputs:
        return {
            "train": train_filtered,
            "test": test,
            "preprocessor": preprocessor,
            "meta": meta,
        }

    processed_dir = Path(processed_dir)
    artifacts_dir = Path(artifacts_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_path, test_path, prep_path, cols_path = _output_paths(
        processed_dir, artifacts_dir, dataset_id
    )

    train_filtered.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    joblib.dump(preprocessor, prep_path)
    cols_path.write_text(json.dumps(meta["feature_columns"], indent=2), encoding="utf-8")

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "preprocessor_path": str(prep_path),
        "feature_columns_path": str(cols_path),
        "meta": meta,
    }
