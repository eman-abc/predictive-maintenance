"""Repository paths for processed data and artifacts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
DEFAULT_DATASET = "FD001"
CMAPSS_DATASETS = ("FD001", "FD002", "FD003", "FD004")


def predictions_path(dataset_id: str = DEFAULT_DATASET) -> Path:
    return PROCESSED / f"cmapss_{dataset_id}_predictions.parquet"


def test_data_path(dataset_id: str = DEFAULT_DATASET) -> Path:
    return PROCESSED / f"cmapss_{dataset_id}_test.parquet"


def phase3_summary_path(dataset_id: str = DEFAULT_DATASET) -> Path:
    return ARTIFACTS / f"cmapss_{dataset_id}_phase3_summary.json"
