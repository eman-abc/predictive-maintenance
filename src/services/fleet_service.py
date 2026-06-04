"""Fleet predictions and telemetry (parquet on disk)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.services.paths import (
    ARTIFACTS,
    CMAPSS_DATASETS,
    DEFAULT_DATASET,
    MODELS,
    predictions_path,
    test_data_path,
    phase3_summary_path,
)


def list_datasets_with_predictions() -> list[str]:
    return [ds for ds in CMAPSS_DATASETS if predictions_path(ds).exists()]


def _normalize_fleet_df(df: pd.DataFrame) -> pd.DataFrame:
    if "failure_prob" in df.columns and "failure_prob_30" not in df.columns:
        df = df.rename(columns={"failure_prob": "failure_prob_30"})
    if "failure_prob_30" in df.columns and "failure_prob" not in df.columns:
        df["failure_prob"] = df["failure_prob_30"]
    return df


def load_fleet_predictions(dataset_id: str = DEFAULT_DATASET) -> pd.DataFrame | None:
    path = predictions_path(dataset_id)
    if not path.exists():
        return None
    return _normalize_fleet_df(pd.read_parquet(path))


def fleet_records(dataset_id: str = DEFAULT_DATASET) -> list[dict[str, Any]]:
    df = load_fleet_predictions(dataset_id)
    if df is None:
        return []
    return _df_to_records(df)


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.replace({float("inf"): None, float("-inf"): None}).to_dict(orient="records")
    for row in out:
        for key, val in list(row.items()):
            if pd.isna(val):
                row[key] = None
            elif hasattr(val, "item"):
                try:
                    row[key] = val.item()
                except (ValueError, AttributeError):
                    pass
    return out


def load_phase3_summary(dataset_id: str = DEFAULT_DATASET) -> dict | None:
    path = phase3_summary_path(dataset_id)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_training_registry() -> dict | None:
    path = ARTIFACTS / "cmapss_training_registry.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_available_models(dataset_id: str = DEFAULT_DATASET) -> list[str]:
    if not MODELS.is_dir():
        return []
    suffix = f"_{dataset_id}"
    return [
        path.stem
        for path in sorted(MODELS.iterdir())
        if path.suffix in {".pkl", ".pt"} and path.stem.endswith(suffix)
    ]


def load_unit_trajectory(unit_id: int, dataset_id: str = DEFAULT_DATASET) -> pd.DataFrame | None:
    path = test_data_path(dataset_id)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df["unit_id"] == unit_id].sort_values("cycle")


def trajectory_records(unit_id: int, dataset_id: str = DEFAULT_DATASET) -> list[dict[str, Any]]:
    traj = load_unit_trajectory(unit_id, dataset_id)
    if traj is None or traj.empty:
        return []
    return _df_to_records(traj)


def asset_row(dataset_id: str, asset_id: str) -> dict[str, Any] | None:
    df = load_fleet_predictions(dataset_id)
    if df is None:
        return None
    match = df[df["asset_id"] == asset_id]
    if match.empty:
        return None
    return _df_to_records(match.iloc[:1])[0]
