"""Load Phase 3 predictions and test trajectories for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
DEFAULT_DATASET = "FD001"
CMAPSS_DATASETS = ("FD001", "FD002", "FD003", "FD004")


def predictions_path(dataset_id: str = DEFAULT_DATASET) -> Path:
    return PROCESSED / f"cmapss_{dataset_id}_predictions.parquet"


def test_data_path(dataset_id: str = DEFAULT_DATASET) -> Path:
    return PROCESSED / f"cmapss_{dataset_id}_test.parquet"


def phase3_summary_path(dataset_id: str = DEFAULT_DATASET) -> Path:
    return ARTIFACTS / f"cmapss_{dataset_id}_phase3_summary.json"


def list_datasets_with_predictions() -> list[str]:
    """FD subsets that have a Phase 3 predictions parquet on disk."""
    return [ds for ds in CMAPSS_DATASETS if predictions_path(ds).exists()]


def render_dataset_selector() -> str:
    """
    Sidebar dataset picker shared across dashboard pages.

    Returns the selected dataset id (e.g. FD001, FD003).
    """
    available = list_datasets_with_predictions()
    if not available:
        st.sidebar.warning(
            "No prediction files found. Run:\n"
            "`python scripts/train_cmapss_phase3.py --dataset FD001`"
        )
        return DEFAULT_DATASET
    default_idx = 0
    if DEFAULT_DATASET in available:
        default_idx = available.index(DEFAULT_DATASET)
    return st.sidebar.selectbox(
        "CMAPSS dataset",
        available,
        index=default_idx,
        help="Switch between trained FD subsets (predictions + summary per dataset).",
    )


def load_fleet_predictions(dataset_id: str = DEFAULT_DATASET) -> pd.DataFrame | None:
    path = predictions_path(dataset_id)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "failure_prob" in df.columns and "failure_prob_30" not in df.columns:
        df = df.rename(columns={"failure_prob": "failure_prob_30"})
    if "failure_prob_30" in df.columns and "failure_prob" not in df.columns:
        df["failure_prob"] = df["failure_prob_30"]
    return df


def load_phase3_summary(dataset_id: str = DEFAULT_DATASET) -> dict | None:
    path = phase3_summary_path(dataset_id)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_unit_trajectory(unit_id: int, dataset_id: str = DEFAULT_DATASET) -> pd.DataFrame | None:
    path = test_data_path(dataset_id)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df["unit_id"] == unit_id].sort_values("cycle")
