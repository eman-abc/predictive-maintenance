"""Databricks MLflow URLs + run discovery (dashboard-local, no src.utils cache issues)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


_load_env()


def databricks_host() -> str | None:
    host = os.getenv("DATABRICKS_HOST", "").strip().rstrip("/")
    return host or None


def is_databricks_tracking() -> bool:
    return os.getenv("MLFLOW_TRACKING_URI", "").strip().lower() == "databricks"


def has_databricks_credentials() -> bool:
    return bool(databricks_host() and os.getenv("DATABRICKS_TOKEN", "").strip())


def experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/predictive_maintenance")


def experiment_id_from_env() -> str | None:
    value = os.getenv("MLFLOW_EXPERIMENT_ID", "").strip()
    return value or None


def resolve_experiment_id(registry: dict | None = None) -> str | None:
    exp_id = experiment_id_from_env()
    if exp_id:
        return exp_id
    if registry:
        reg_id = registry.get("mlflow_experiment_id")
        if reg_id:
            return str(reg_id)
    return None


def pkl_bundle_run_id_from_env() -> str | None:
    value = os.getenv("MLFLOW_PKL_BUNDLE_RUN_ID", "").strip()
    return value or None


def pkl_bundle_run_name_from_env() -> str:
    return os.getenv("MLFLOW_PKL_BUNDLE_RUN_NAME", "cmapss_pkl_bundle_v2").strip()


def run_url(experiment_id: str, run_id: str, *, section: str = "overview") -> str | None:
    host = databricks_host()
    if not host or not experiment_id or not run_id:
        return None
    if section == "artifacts":
        return f"{host}/ml/experiments/{experiment_id}/runs/{run_id}/artifactPath/models"
    if section == "metrics":
        return f"{host}/ml/experiments/{experiment_id}/runs/{run_id}/model-metrics"
    return f"{host}/ml/experiments/{experiment_id}/runs/{run_id}/overview"


def experiment_url(experiment_id: str) -> str | None:
    host = databricks_host()
    if not host or not experiment_id:
        return None
    return f"{host}/ml/experiments/{experiment_id}"


def _registry_has_run_ids(registry: dict) -> bool:
    for entry in (registry.get("datasets") or {}).values():
        if entry.get("mlflow_run_id"):
            return True
    return False


def can_build_registry_links(registry: dict | None) -> bool:
    if not databricks_host() or not registry:
        return False
    if not _registry_has_run_ids(registry):
        return False
    return resolve_experiment_id(registry) is not None


def registry_bundle_row(experiment_id: str) -> dict[str, Any] | None:
    run_id = pkl_bundle_run_id_from_env()
    if not run_id:
        return None
    name = pkl_bundle_run_name_from_env()
    return {
        "run_name": name,
        "run_id": run_id,
        "run_url": run_url(experiment_id, run_id),
        "artifacts_url": run_url(experiment_id, run_id, section="artifacts"),
    }


def _run_display_name(row: Any) -> str:
    return str(row.get("tags.mlflow.runName") or row.get("run_name") or "")


def _metric(row: Any, *keys: str) -> Any:
    for key in keys:
        if key in row.index and row.get(key) is not None:
            return row.get(key)
    return None


def get_experiment_and_runs(*, max_results: int = 200):
    import mlflow
    from mlflow.entities import ViewType

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    exp_id_override = os.getenv("MLFLOW_EXPERIMENT_ID", "").strip()
    if exp_id_override:
        exp = mlflow.get_experiment(exp_id_override)
    else:
        exp = mlflow.get_experiment_by_name(experiment_name())
    if exp is None:
        return None, None

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=["start_time DESC"],
    )
    return exp, runs


def phase3_summary_rows(runs, experiment_id: str) -> list[dict[str, Any]]:
    if runs is None or runs.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, row in runs.iterrows():
        name = _run_display_name(row)
        if not name.endswith("_phase3_summary"):
            continue
        run_id = str(row.get("run_id", ""))
        dataset_id = row.get("params.dataset_id") or name.replace("_phase3_summary", "")
        rows.append(
            {
                "dataset": str(dataset_id),
                "run_name": name,
                "run_id": run_id,
                "start_time": str(row.get("start_time", "")),
                "winner": row.get("params.winner"),
                "test_rmse": _metric(row, "metrics.test_rmse"),
                "test_nasa": _metric(row, "metrics.test_rul_score", "metrics.test_nasa_score"),
                "test_cox_rmse": _metric(row, "metrics.test_cox_rmse"),
                "cox_concordance": _metric(row, "metrics.val_concordance", "metrics.cox_concordance"),
                "training_batch": row.get("params.training_batch") or row.get("tags.training_batch"),
                "duration_min": _metric(row, "metrics.duration_min"),
                "run_url": run_url(experiment_id, run_id),
                "artifacts_url": run_url(experiment_id, run_id, section="artifacts"),
            }
        )

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        ds = row["dataset"]
        if ds not in latest or row.get("start_time", "") > latest[ds].get("start_time", ""):
            latest[ds] = row
    return sorted(latest.values(), key=lambda r: r["dataset"])


def artifact_bundle_rows(runs, experiment_id: str) -> list[dict[str, Any]]:
    if runs is None or runs.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, row in runs.iterrows():
        name = _run_display_name(row)
        if "pkl_bundle" not in name.lower() and "pkl" not in name.lower():
            continue
        run_id = str(row.get("run_id", ""))
        rows.append(
            {
                "run_name": name,
                "run_id": run_id,
                "start_time": str(row.get("start_time", "")),
                "run_url": run_url(experiment_id, run_id),
                "artifacts_url": run_url(experiment_id, run_id, section="artifacts"),
            }
        )
    if not rows:
        return []
    latest = max(rows, key=lambda r: r.get("start_time", ""))
    return [latest]


def registry_fallback_rows(registry: dict, experiment_id: str | None) -> list[dict[str, Any]]:
    if not experiment_id:
        return []
    rows: list[dict[str, Any]] = []
    for ds_id, entry in (registry.get("datasets") or {}).items():
        run_id = entry.get("mlflow_run_id")
        if not run_id:
            continue
        name = entry.get("mlflow_run_name") or f"{ds_id}_phase3_summary"
        rows.append(
            {
                "dataset": ds_id,
                "run_name": name,
                "run_id": run_id,
                "winner": entry.get("winner"),
                "test_rmse": entry.get("test_rmse"),
                "test_nasa": entry.get("test_nasa_score"),
                "test_cox_rmse": entry.get("test_cox_rmse"),
                "cox_concordance": None,
                "training_batch": entry.get("training_batch"),
                "duration_min": None,
                "run_url": run_url(experiment_id, str(run_id)),
                "artifacts_url": run_url(experiment_id, str(run_id), section="artifacts"),
            }
        )
    return sorted(rows, key=lambda r: r["dataset"])


def collect_databricks_run_links(registry: dict | None = None) -> dict[str, Any]:
    exp_id = resolve_experiment_id(registry)
    out: dict[str, Any] = {
        "experiment_name": experiment_name(),
        "experiment_id": exp_id,
        "experiment_url": experiment_url(exp_id) if exp_id else None,
        "phase3_rows": [],
        "bundle_rows": [],
        "source": "none",
        "error": None,
        "registry_only": False,
    }

    exp = None
    runs = None
    if has_databricks_credentials():
        try:
            prev = os.getenv("MLFLOW_TRACKING_URI")
            if not prev or prev == "./mlruns":
                os.environ["MLFLOW_TRACKING_URI"] = "databricks"
            exp, runs = get_experiment_and_runs()
        except Exception as exc:
            out["error"] = str(exc)

    if exp is not None:
        out["experiment_id"] = exp.experiment_id
        out["experiment_url"] = experiment_url(exp.experiment_id)
        out["phase3_rows"] = phase3_summary_rows(runs, exp.experiment_id)
        out["bundle_rows"] = artifact_bundle_rows(runs, exp.experiment_id)
        out["source"] = "live"

    exp_id = out["experiment_id"]
    if exp_id:
        out["experiment_url"] = experiment_url(exp_id)

    if not out["phase3_rows"] and registry and exp_id:
        out["phase3_rows"] = registry_fallback_rows(registry, exp_id)
        out["source"] = "registry"
        out["registry_only"] = True

    if not out["bundle_rows"] and exp_id:
        bundle = registry_bundle_row(exp_id)
        if bundle:
            out["bundle_rows"] = [bundle]
            if out["source"] == "registry":
                out["source"] = "registry+env"

    return out
