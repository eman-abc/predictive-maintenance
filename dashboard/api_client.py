"""HTTP client for the deployment FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

_DEFAULT_TIMEOUT = 120.0
_CMMS_TIMEOUT = float(os.getenv("API_CMMS_TIMEOUT_SECONDS", "35"))
_METRICS_TIMEOUT = float(os.getenv("API_METRICS_TIMEOUT_SECONDS", "90"))


def api_base_url() -> str | None:
    url = (os.getenv("API_BASE_URL") or "").strip().rstrip("/")
    return url or None


def use_api_backend() -> bool:
    return api_base_url() is not None


def _client(*, timeout: float | None = None) -> httpx.Client:
    base = api_base_url()
    if not base:
        raise RuntimeError("API_BASE_URL is not set")
    return httpx.Client(base_url=base, timeout=timeout or _DEFAULT_TIMEOUT)


def _get(path: str, *, timeout: float | None = None, **params: Any) -> Any:
    with _client(timeout=timeout) as client:
        resp = client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()


def _post(path: str, json: dict[str, Any], *, timeout: float | None = None) -> Any:
    with _client(timeout=timeout) as client:
        resp = client.post(path, json=json)
        resp.raise_for_status()
        return resp.json()


@st.cache_data(ttl=30)
def get_health() -> dict[str, Any]:
    return _get("/health")


@st.cache_data(ttl=60)
def list_datasets() -> list[str]:
    return _get("/datasets")["datasets"]


@st.cache_data(ttl=30)
def get_fleet_df(dataset_id: str) -> pd.DataFrame | None:
    try:
        data = _get("/fleet", dataset=dataset_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise
    rows = data.get("rows") or []
    if not rows:
        return None
    return pd.DataFrame(rows)


def get_phase3_summary(dataset_id: str) -> dict | None:
    try:
        data = _get("/metrics/phase3", dataset=dataset_id)
        return data.get("summary")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise


@st.cache_data(ttl=120)
def get_training_registry() -> dict | None:
    try:
        return _get("/metrics/registry")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise


def list_available_models(dataset_id: str) -> list[str]:
    try:
        return _get("/metrics/models", dataset=dataset_id).get("models") or []
    except httpx.HTTPStatusError:
        return []


def get_asset_bundle(dataset_id: str, asset_id: str) -> dict[str, Any] | None:
    try:
        return _get(f"/assets/{asset_id}", dataset=dataset_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise


def get_alerts_df(dataset_id: str, levels: list[str]) -> pd.DataFrame:
    level_param = ",".join(levels)
    data = _get("/alerts", dataset=dataset_id, level=level_param)
    rows = data.get("rows") or []
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def post_alert_ack(dataset_id: str, asset_id: str, alert_level: str) -> dict[str, Any]:
    return _post(
        "/alerts/ack",
        {
            "dataset_id": dataset_id,
            "asset_id": asset_id,
            "alert_level": alert_level,
        },
    )


def post_auto_dispatch(dataset_id: str, levels: list[str] | None = None) -> dict[str, Any]:
    return _post(
        "/cmms/auto-dispatch",
        {
            "dataset_id": dataset_id,
            "levels": levels or ["critical"],
        },
        timeout=_CMMS_TIMEOUT,
    )


def post_shift_briefing(
    *,
    mode: str,
    dataset_id: str,
    levels: list[str],
) -> dict[str, Any]:
    return _post(
        "/briefings/shift",
        {"mode": mode, "dataset_id": dataset_id, "levels": levels},
    )


def get_recent_auto_work_orders(limit: int = 20) -> list[dict[str, Any]]:
    data = _get("/cmms/workorders/recent/auto", limit=limit)
    return data.get("rows") or []


def post_briefing(*, mode: str, dataset_id: str, context: dict[str, Any]) -> dict[str, Any]:
    return _post(
        "/briefings",
        {"mode": mode, "dataset_id": dataset_id, "context": context},
    )


def post_metric_explain(
    *,
    section_id: str,
    dataset_id: str,
    summary: dict,
    registry_entry: dict | None,
) -> dict[str, Any]:
    return _post(
        "/metrics/explain",
        {
            "section_id": section_id,
            "dataset_id": dataset_id,
            "summary": summary,
            "registry_entry": registry_entry,
        },
        timeout=_METRICS_TIMEOUT,
    )


def post_work_orders(dataset_id: str, levels: list[str]) -> dict[str, Any]:
    return _post(
        "/cmms/workorders",
        {"dataset_id": dataset_id, "levels": levels},
    )


def get_cmms_databricks_status() -> dict[str, Any]:
    return _get("/cmms/databricks/status")


def get_recent_work_orders(limit: int = 20) -> list[dict[str, Any]]:
    data = _get("/cmms/workorders/recent", limit=limit)
    return data.get("rows") or []


def get_mlflow_links(dataset_id: str) -> dict[str, Any]:
    return _get("/mlflow/links", dataset=dataset_id)
