"""MLflow / Databricks link helpers for the API."""

from __future__ import annotations

from typing import Any


def mlflow_panel_data(registry: dict | None, dataset_id: str) -> dict[str, Any]:
    from dashboard import mlflow_links

    exp_id = mlflow_links.resolve_experiment_id(registry)
    host = mlflow_links.databricks_host()
    return {
        "databricks_host": host,
        "experiment_id": exp_id,
        "experiment_name": mlflow_links.experiment_name(),
        "experiment_url": mlflow_links.experiment_url(exp_id) if exp_id and host else None,
        "pkl_bundle_run_id": mlflow_links.pkl_bundle_run_id_from_env(),
        "pkl_bundle_url": mlflow_links.pkl_bundle_artifact_url(),
        "dataset_id": dataset_id,
        "tracking_uri": mlflow_links.is_databricks_tracking(),
    }
