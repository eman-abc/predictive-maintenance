"""Load Phase 3 predictions — via API (deployment) or local parquet."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.api_client import use_api_backend

ROOT = Path(__file__).resolve().parents[1]
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


def list_datasets_with_predictions() -> list[str]:
    if use_api_backend():
        from dashboard import api_client

        return api_client.list_datasets()
    return [ds for ds in CMAPSS_DATASETS if predictions_path(ds).exists()]


def render_dataset_selector() -> str:
    available = list_datasets_with_predictions()
    if not available:
        st.sidebar.warning(
            "No prediction files found. Run:\n"
            "`python scripts/import_cmapss_colab_outputs.py --force`"
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
    if use_api_backend():
        from dashboard import api_client

        return api_client.get_fleet_df(dataset_id)
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
    if use_api_backend():
        from dashboard import api_client

        return api_client.get_phase3_summary(dataset_id)
    path = phase3_summary_path(dataset_id)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_training_registry() -> dict | None:
    if use_api_backend():
        from dashboard import api_client

        return api_client.get_training_registry()
    path = ARTIFACTS / "cmapss_training_registry.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_available_models(dataset_id: str = DEFAULT_DATASET) -> list[str]:
    if use_api_backend():
        from dashboard import api_client

        return api_client.list_available_models(dataset_id)
    if not MODELS.is_dir():
        return []
    suffix = f"_{dataset_id}"
    return [
        path.stem
        for path in sorted(MODELS.iterdir())
        if path.suffix in {".pkl", ".pt"} and path.stem.endswith(suffix)
    ]


def load_unit_trajectory(unit_id: int, dataset_id: str = DEFAULT_DATASET) -> pd.DataFrame | None:
    if use_api_backend():
        from dashboard import api_client

        asset_id = f"ENG-{unit_id:03d}"
        bundle = api_client.get_asset_bundle(dataset_id, asset_id)
        if not bundle or not bundle.get("trajectory"):
            return None
        return pd.DataFrame(bundle["trajectory"])
    path = test_data_path(dataset_id)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df["unit_id"] == unit_id].sort_values("cycle")


def survival_model_path(dataset_id: str = DEFAULT_DATASET) -> Path:
    return MODELS / f"survival_{dataset_id}.pkl"


def load_survival_model(dataset_id: str = DEFAULT_DATASET):
    if use_api_backend():
        return None
    path = survival_model_path(dataset_id)
    if not path.exists():
        return None
    from src.models.survival_model import SurvivalModel

    return SurvivalModel.load(path)
