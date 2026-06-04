"""Append CMMS work-order events to a Unity Catalog Delta table in Databricks."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from src.alerts.alert_generator import Alert
from src.alerts.cmms_routing import map_escalation

load_dotenv()

_FQN_RE = re.compile(r"^[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+$")


def _enabled() -> bool:
    return os.getenv("CMMS_LOG_TO_DATABRICKS", "false").lower() in (
        "1",
        "true",
        "yes",
    )


def _fix_fqn_if_catalog_missing(fqn: str) -> str:
    """Use a visible UC catalog when CMMS_DELTA_TABLE points at e.g. main (not in workspace)."""
    parts = fqn.split(".")
    if len(parts) != 3:
        return fqn
    catalog, schema, table = parts
    try:
        host, token, _ = databricks_credentials()
        from src.utils.databricks_uc import list_uc_catalogs, pick_cmms_catalog

        catalogs = list_uc_catalogs(host, token)
        if catalog in catalogs:
            return fqn
        chosen = pick_cmms_catalog(
            catalogs,
            preferred=(os.getenv("CMMS_DELTA_CATALOG") or "").strip() or None,
        )
        if not chosen:
            return fqn
        fixed = f"{chosen}.{schema}.{table}"
        import warnings

        warnings.warn(
            f"CMMS_DELTA_TABLE catalog {catalog!r} not found in workspace "
            f"({catalogs}). Using {fixed!r} instead. Update .env to avoid this warning.",
            stacklevel=2,
        )
        return fixed
    except (ValueError, OSError):
        return fqn


def table_fqn() -> str:
    """Fully qualified table name: catalog.schema.table."""
    explicit = (os.getenv("CMMS_DELTA_TABLE") or "").strip()
    if explicit:
        if not _FQN_RE.match(explicit):
            raise ValueError(
                f"CMMS_DELTA_TABLE must be catalog.schema.table, got {explicit!r}"
            )
        return _fix_fqn_if_catalog_missing(explicit)
    catalog = (os.getenv("CMMS_DELTA_CATALOG") or os.getenv("MLFLOW_UC_CATALOG") or "").strip()
    schema = (os.getenv("CMMS_DELTA_SCHEMA") or os.getenv("MLFLOW_UC_SCHEMA") or "cmapss").strip()
    table = (os.getenv("CMMS_DELTA_TABLE_NAME") or "cmms_work_orders").strip()
    if not catalog:
        try:
            host, token, _ = databricks_credentials()
            from src.utils.databricks_uc import pick_cmms_catalog, list_uc_catalogs

            catalog = pick_cmms_catalog(list_uc_catalogs(host, token)) or ""
        except ValueError:
            catalog = ""
    if not catalog:
        raise ValueError(
            "Set CMMS_DELTA_TABLE=catalog.schema.cmms_work_orders or CMMS_DELTA_CATALOG + schema. "
            "Run: python scripts/diagnose_databricks_registry.py"
        )
    fqn = f"{catalog}.{schema}.{table}"
    if not _FQN_RE.match(fqn):
        raise ValueError(f"Invalid CMMS Delta table name: {fqn!r}")
    return fqn


def sql_http_path() -> str:
    """SQL warehouse HTTP path for databricks-sql-connector."""
    path = (os.getenv("DATABRICKS_SQL_HTTP_PATH") or "").strip()
    if path:
        return path if path.startswith("/") else f"/{path}"
    warehouse_id = (os.getenv("DATABRICKS_SQL_WAREHOUSE_ID") or "").strip()
    if warehouse_id:
        return f"/sql/1.0/warehouses/{warehouse_id}"
    raise ValueError(
        "Set DATABRICKS_SQL_HTTP_PATH or DATABRICKS_SQL_WAREHOUSE_ID "
        "(SQL → SQL Warehouses → Connection details)."
    )


def databricks_credentials() -> tuple[str, str, str]:
    host = (os.getenv("DATABRICKS_HOST") or "").strip().rstrip("/")
    token = (os.getenv("DATABRICKS_TOKEN") or "").strip()
    if not host or not token:
        raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN are required.")
    if not host.startswith("https://"):
        host = f"https://{host}"
    return host, token, sql_http_path()


def is_databricks_logging_configured() -> bool:
    if not _enabled():
        return False
    try:
        databricks_credentials()
        table_fqn()
        return True
    except ValueError:
        return False


def explore_table_url(fqn: str | None = None) -> str | None:
    """Link to Catalog Explorer (best-effort; workspace UI may vary)."""
    links = databricks_table_links(fqn)
    return links.get("explore_url")


def databricks_table_links(fqn: str | None = None) -> dict[str, str]:
    """URLs and query text for viewing work orders after submit."""
    try:
        host, _, http_path = databricks_credentials()
    except ValueError:
        return {}
    fqn = fqn or table_fqn()
    parts = fqn.split(".")
    if len(parts) != 3:
        return {}
    catalog, schema, table = parts
    explore = f"{host}/explore/data/{catalog}/{schema}/{table}"
    # SQL editor entry point (warehouse from connection path)
    warehouse_id = ""
    if "/warehouses/" in http_path:
        warehouse_id = http_path.rstrip("/").split("/")[-1]
    sql_editor = f"{host}/sql/warehouses/{warehouse_id}" if warehouse_id else f"{host}/sql"
    sample_query = (
        f"SELECT work_order_id, asset_id, escalation_tier, cmms_priority, "
        f"sla_label, submit_status, submitted_at\n"
        f"FROM {fqn}\n"
        f"ORDER BY submitted_at DESC\n"
        f"LIMIT 50;"
    )
    return {
        "explore_url": explore,
        "sql_editor_url": sql_editor,
        "table_fqn": fqn,
        "sample_query": sample_query,
    }


def schema_fqn_from_table(fqn: str | None = None) -> str:
    """Return catalog.schema from a three-part table name."""
    parts = (fqn or table_fqn()).split(".")
    if len(parts) != 3:
        raise ValueError(f"Expected catalog.schema.table, got {fqn!r}")
    return f"{parts[0]}.{parts[1]}"


def create_schema_sql(fqn: str | None = None) -> str:
    schema = schema_fqn_from_table(fqn)
    return f"""
CREATE SCHEMA IF NOT EXISTS {schema}
COMMENT 'UC5 predictive maintenance CMMS work order audit schema'
""".strip()


def ensure_schema(*, fqn: str | None = None) -> None:
    """Create Unity Catalog schema if missing (e.g. main.cmapss)."""
    fqn = fqn or table_fqn()
    sql = create_schema_sql(fqn)
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)


def create_table_sql(fqn: str | None = None) -> str:
    t = fqn or table_fqn()
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
  work_order_id STRING NOT NULL,
  alert_id STRING,
  asset_id STRING NOT NULL,
  dataset_id STRING,
  submit_status STRING,
  priority STRING,
  cmms_priority STRING,
  escalation_tier STRING,
  sla_response_hours INT,
  sla_label STRING,
  alert_level STRING,
  risk_score DOUBLE,
  time_to_failure_cycles DOUBLE,
  predicted_rul DOUBLE,
  failure_probability_30 DOUBLE,
  failure_probability_72 DOUBLE,
  anomaly_score DOUBLE,
  recommended_action STRING,
  description STRING,
  sensor_readings_json STRING,
  payload_json STRING,
  cmms_response_json STRING,
  submitted_at TIMESTAMP,
  source STRING
)
USING DELTA
COMMENT 'UC5 predictive maintenance — CMMS work order audit log (mock + production adapter)'
""".strip()


def _connect():
    """Open a Databricks SQL connection (requires ``databricks-sql-connector``)."""
    try:
        from databricks import sql
    except ImportError as exc:
        raise ImportError(
            "Install databricks-sql-connector: pip install databricks-sql-connector"
        ) from exc

    host, token, http_path = databricks_credentials()
    hostname = host.replace("https://", "").replace("http://", "")
    return sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=token,
    )


def _migrate_table_columns(fqn: str) -> None:
    """Add SLA columns to existing tables (no-op if already present)."""
    alters = [
        f"ALTER TABLE {fqn} ADD COLUMN IF NOT EXISTS cmms_priority STRING",
        f"ALTER TABLE {fqn} ADD COLUMN IF NOT EXISTS sla_response_hours INT",
        f"ALTER TABLE {fqn} ADD COLUMN IF NOT EXISTS sla_label STRING",
    ]
    with _connect() as conn:
        with conn.cursor() as cursor:
            for stmt in alters:
                try:
                    cursor.execute(stmt)
                except Exception:
                    pass


def ensure_table(*, fqn: str | None = None) -> None:
    """Create schema and Delta table if missing."""
    fqn = fqn or table_fqn()
    ensure_schema(fqn=fqn)
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(create_table_sql(fqn))
    _migrate_table_columns(fqn)


def _alert_to_row(
    alert: Alert,
    *,
    submit_status: str,
    payload: dict[str, Any],
    dataset_id: str | None = None,
    cmms_response: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    meta = alert.metadata or {}
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    work_order_id = f"WO-{alert.alert_id.replace('ALT-', '')}"
    sensors = meta.get("sensor_readings") or {}
    routing = map_escalation(
        payload.get("escalation_tier") or meta.get("escalation_tier"),
        alert_level=alert.level,
    )
    cmms_priority = payload.get("cmms_priority") or routing.cmms_priority
    sla_hours = int(payload.get("sla_response_hours") or routing.sla_response_hours)
    sla_label = str(payload.get("sla_label") or routing.sla_label)
    escalation = str(payload.get("escalation_tier") or routing.escalation_tier)
    legacy_priority = payload.get("priority_legacy") or (
        "high" if alert.level.value == "critical" else "medium"
    )
    return (
        work_order_id,
        alert.alert_id,
        alert.asset_id,
        dataset_id or meta.get("dataset_id"),
        submit_status,
        legacy_priority,
        cmms_priority,
        escalation,
        sla_hours,
        sla_label,
        alert.level.value,
        float(meta.get("risk_score", alert.health_score)),
        float(meta.get("time_to_failure_cycles", alert.rul)),
        float(alert.rul),
        float(alert.failure_probability),
        float(meta.get("failure_prob_72", 0)),
        float(meta.get("anomaly_score", 0)),
        str(meta.get("recommended_action", alert.description)),
        alert.description,
        json.dumps(sensors),
        json.dumps(payload),
        json.dumps(cmms_response) if cmms_response else None,
        now,
        os.getenv("CMMS_SOURCE", "streamlit_dashboard"),
    )


def insert_work_order(
    alert: Alert,
    *,
    submit_status: str,
    payload: dict[str, Any],
    dataset_id: str | None = None,
    cmms_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert one work-order row into the Delta table."""
    if not _enabled():
        return {"status": "skipped", "reason": "CMMS_LOG_TO_DATABRICKS is false"}

    fqn = table_fqn()
    ensure_table(fqn=fqn)

    row = _alert_to_row(
        alert,
        submit_status=submit_status,
        payload=payload,
        dataset_id=dataset_id,
        cmms_response=cmms_response,
    )
    # row includes cmms_response at end — fix tuple order in INSERT

    insert_sql = f"""
INSERT INTO {fqn} (
  work_order_id, alert_id, asset_id, dataset_id, submit_status, priority,
  cmms_priority, escalation_tier, sla_response_hours, sla_label, alert_level,
  risk_score, time_to_failure_cycles, predicted_rul, failure_probability_30,
  failure_probability_72, anomaly_score, recommended_action, description,
  sensor_readings_json, payload_json, cmms_response_json, submitted_at, source
) VALUES (
  ?, ?, ?, ?, ?, ?,
  ?, ?, ?, ?, ?,
  ?, ?, ?, ?,
  ?, ?, ?,
  ?, ?, ?, ?,
  ?, ?
)
"""
    params = row

    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(insert_sql, params)

    links = databricks_table_links(fqn)
    return {
        "status": "databricks_logged",
        "table": fqn,
        "work_order_id": row[0],
        **links,
    }


def fetch_recent_work_orders(*, limit: int = 25) -> list[dict[str, Any]]:
    """Return recent rows for dashboard preview."""
    fqn = table_fqn()
    query = f"""
SELECT work_order_id, asset_id, dataset_id, submit_status, alert_level,
       escalation_tier, cmms_priority, sla_label, risk_score,
       time_to_failure_cycles, submitted_at
FROM {fqn}
ORDER BY submitted_at DESC
LIMIT {int(limit)}
"""
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
