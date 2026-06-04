"""Auto-dispatch critical alerts to a separate Databricks Delta table."""

from __future__ import annotations

import os
from typing import Any

from src.alerts.alert_generator import Alert
from src.alerts.cmms_databricks import (
    auto_table_fqn,
    databricks_table_links,
    ensure_table,
    fetch_asset_ids_in_table,
    insert_work_order,
    is_databricks_logging_configured,
)
from src.alerts.cmms_routing import map_escalation, routing_to_payload_fields
from src.services.alerts_service import build_alerts_for_cmms


def auto_dispatch_enabled() -> bool:
    return os.getenv("CMMS_AUTO_DISPATCH", "true").lower() in ("1", "true", "yes")


def auto_dispatch_critical(
    dataset_id: str,
    *,
    levels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Write critical (default) alerts to CMMS_AUTO_DELTA_TABLE without operator click.
    Skips assets already present with submit_status=auto_dispatched.
    """
    levels = levels or ["critical"]
    if "critical" not in [x.lower() for x in levels]:
        levels = ["critical"]

    if not auto_dispatch_enabled():
        return {"status": "skipped", "reason": "CMMS_AUTO_DISPATCH is false"}
    if not is_databricks_logging_configured():
        return {"status": "skipped", "reason": "Databricks logging not configured"}

    fqn = auto_table_fqn()
    ensure_table(fqn=fqn)
    existing = fetch_asset_ids_in_table(
        dataset_id=dataset_id,
        fqn=fqn,
        submit_status="auto_dispatched",
    )

    dispatched: list[dict[str, Any]] = []
    skipped_assets: list[str] = []

    for alert, alert_id in build_alerts_for_cmms(dataset_id, levels=levels):
        if alert.asset_id in existing:
            skipped_assets.append(alert.asset_id)
            continue
        routing = map_escalation(
            (alert.metadata or {}).get("escalation_tier"),
            alert_level=alert.level,
        )
        payload = {
            **routing_to_payload_fields(routing),
            "work_order_id": f"WO-AUTO-{dataset_id}-{alert.asset_id}",
            "source": "auto_pipeline",
            "priority": routing.cmms_priority,
            "priority_legacy": "high",
        }
        db = insert_work_order(
            alert,
            submit_status="auto_dispatched",
            payload=payload,
            dataset_id=dataset_id,
            cmms_response={"status": "auto_pipeline"},
            table=fqn,
        )
        dispatched.append(db)
        existing.add(alert.asset_id)

    links = databricks_table_links(fqn)
    return {
        "status": "ok",
        "table": fqn,
        "dispatched_count": len(dispatched),
        "skipped_count": len(skipped_assets),
        "skipped_assets": skipped_assets,
        "results": dispatched,
        **links,
    }
