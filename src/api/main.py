"""FastAPI application — deployment backend."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.alerts.cmms_databricks import (
    auto_table_fqn,
    databricks_status_payload,
    fetch_recent_work_orders,
    is_databricks_logging_configured,
)
from src.alerts.cmms_mock import CMMSClient
from src.api.schemas import (
    AlertAckRequest,
    AutoDispatchRequest,
    BriefingRequest,
    BriefingResponse,
    HealthResponse,
    MetricExplainRequest,
    MetricExplainResponse,
    ShiftBriefingRequest,
    ShiftBriefingResponse,
    WorkOrdersRequest,
    WorkOrdersResponse,
)
from src.services import alert_ack_store, alerts_service, briefing_service, fleet_service
from src.services.cmms_auto_dispatch import auto_dispatch_critical
from src.services.mlflow_links_service import mlflow_panel_data

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("OLLAMA_PRELOAD_ON_STARTUP", "true").lower() in ("1", "true", "yes"):
        briefing_service.preload_ollama()
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description="UC5 deployment backend — fleet, alerts, briefings, CMMS",
    version="1.0.0",
    lifespan=lifespan,
)

_cors = os.getenv("API_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors.split(",")] if _cors != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ollama = briefing_service.ollama_status()
    return HealthResponse(
        status="ok",
        ollama_available=ollama["available"],
        ollama_model=ollama.get("model"),
        datasets_available=fleet_service.list_datasets_with_predictions(),
    )


@app.get("/datasets")
def list_datasets() -> dict[str, list[str]]:
    return {"datasets": fleet_service.list_datasets_with_predictions()}


@app.get("/fleet")
def get_fleet(dataset: str = Query("FD001")) -> dict[str, Any]:
    records = fleet_service.fleet_records(dataset)
    if not records:
        raise HTTPException(404, f"No predictions for dataset {dataset}")
    return {"dataset_id": dataset, "count": len(records), "rows": records}


@app.get("/assets/{asset_id}")
def get_asset(
    asset_id: str,
    dataset: str = Query("FD001"),
) -> dict[str, Any]:
    row = fleet_service.asset_row(dataset, asset_id)
    if row is None:
        raise HTTPException(404, f"Asset {asset_id} not found in {dataset}")
    try:
        unit_id = int(str(asset_id).split("-")[1])
    except (IndexError, ValueError) as exc:
        raise HTTPException(400, "Invalid asset_id format") from exc
    trajectory = fleet_service.trajectory_records(unit_id, dataset)
    return {
        "dataset_id": dataset,
        "asset_id": asset_id,
        "row": row,
        "trajectory": trajectory,
    }


@app.get("/alerts")
def get_alerts(
    dataset: str = Query("FD001"),
    level: str | None = Query(None, description="Comma-separated: critical,warning"),
) -> dict[str, Any]:
    levels = [x.strip() for x in level.split(",")] if level else ["critical", "warning"]
    rows = alerts_service.list_alert_rows(dataset, levels=levels)
    rows = alert_ack_store.enrich_rows_with_ack(rows, dataset_id=dataset)
    return {"dataset_id": dataset, "count": len(rows), "rows": rows}


@app.post("/alerts/ack")
def acknowledge_alert(body: AlertAckRequest) -> dict[str, str]:
    alert_ack_store.acknowledge(body.dataset_id, body.asset_id, body.alert_level)
    return {"status": "acknowledged", "asset_id": body.asset_id}


@app.get("/metrics/phase3")
def get_phase3_summary(dataset: str = Query("FD001")) -> dict[str, Any]:
    summary = fleet_service.load_phase3_summary(dataset)
    if summary is None:
        raise HTTPException(404, f"No phase3 summary for {dataset}")
    return {"dataset_id": dataset, "summary": summary}


@app.get("/metrics/registry")
def get_registry() -> dict[str, Any]:
    registry = fleet_service.load_training_registry()
    if registry is None:
        raise HTTPException(404, "Training registry not found")
    return registry


@app.get("/metrics/models")
def list_models(dataset: str = Query("FD001")) -> dict[str, list[str]]:
    return {
        "dataset_id": dataset,
        "models": fleet_service.list_available_models(dataset),
    }


@app.get("/mlflow/links")
def mlflow_links(dataset: str = Query("FD001")) -> dict[str, Any]:
    registry = fleet_service.load_training_registry()
    return mlflow_panel_data(registry, dataset)


@app.post("/briefings", response_model=BriefingResponse)
def create_briefing(body: BriefingRequest) -> BriefingResponse:
    result = briefing_service.generate_briefing(mode=body.mode, context=body.context)
    return BriefingResponse(**result)


@app.post("/briefings/shift", response_model=ShiftBriefingResponse)
def create_shift_briefing(body: ShiftBriefingRequest) -> ShiftBriefingResponse:
    result = briefing_service.generate_shift_briefing(
        mode=body.mode,
        dataset_id=body.dataset_id,
        levels=body.levels,
    )
    return ShiftBriefingResponse(**result)


@app.post("/metrics/explain", response_model=MetricExplainResponse)
def explain_metrics(body: MetricExplainRequest) -> MetricExplainResponse:
    result = briefing_service.generate_metric_explanation(
        section_id=body.section_id,
        dataset_id=body.dataset_id,
        summary=body.summary,
        registry_entry=body.registry_entry,
    )
    return MetricExplainResponse(**result)


@app.post("/cmms/workorders", response_model=WorkOrdersResponse)
def submit_work_orders(body: WorkOrdersRequest) -> WorkOrdersResponse:
    client = CMMSClient()
    results: list[dict[str, Any]] = []
    for alert, _aid in alerts_service.build_alerts_for_cmms(
        body.dataset_id, levels=body.levels
    ):
        results.append(client.create_work_order(alert, dataset_id=body.dataset_id))
    return WorkOrdersResponse(results=results, count=len(results))


@app.post("/cmms/auto-dispatch")
def cmms_auto_dispatch(body: AutoDispatchRequest) -> dict[str, Any]:
    return auto_dispatch_critical(body.dataset_id, levels=body.levels)


@app.get("/cmms/databricks/status")
def cmms_databricks_status() -> dict[str, Any]:
    return databricks_status_payload()


@app.get("/cmms/workorders/recent")
def recent_work_orders(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    if not is_databricks_logging_configured():
        return {"rows": [], "configured": False}
    try:
        rows = fetch_recent_work_orders(limit=limit)
        return {"rows": rows, "configured": True}
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/cmms/workorders/recent/auto")
def recent_auto_work_orders(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    if not is_databricks_logging_configured():
        return {"rows": [], "configured": False}
    try:
        rows = fetch_recent_work_orders(limit=limit, fqn=auto_table_fqn())
        return {"rows": rows, "configured": True, "table": auto_table_fqn()}
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc
