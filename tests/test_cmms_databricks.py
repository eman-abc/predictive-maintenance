"""Tests for Databricks CMMS work-order logging."""

from unittest.mock import MagicMock, patch

import pytest

from src.alerts.alert_generator import Alert
from src.alerts.cmms_databricks import (
    create_schema_sql,
    create_table_sql,
    insert_work_order,
    is_databricks_logging_configured,
    schema_fqn_from_table,
    table_fqn,
)
from src.alerts.cmms_mock import CMMSClient
from src.alerts.threshold_engine import AlertLevel


@pytest.fixture
def sample_alert() -> Alert:
    return Alert(
        alert_id="ALT-000042",
        asset_id="ENG-042",
        level=AlertLevel.CRITICAL,
        title="Critical",
        description="High risk",
        health_score=72.0,
        rul=12.0,
        failure_probability=0.88,
        metadata={
            "escalation_tier": "L2-Critical",
            "recommended_action": "Inspect bearings",
            "sensor_readings": {"sensor_2": 600.0},
            "risk_score": 72.0,
            "time_to_failure_cycles": 12.0,
            "failure_prob_72": 0.91,
            "anomaly_score": 61.0,
        },
    )


def test_table_fqn_from_parts(monkeypatch):
    monkeypatch.setenv("CMMS_DELTA_CATALOG", "workspace")
    monkeypatch.setenv("CMMS_DELTA_SCHEMA", "cmapss")
    monkeypatch.setenv("CMMS_DELTA_TABLE_NAME", "cmms_work_orders")
    assert table_fqn() == "workspace.cmapss.cmms_work_orders"


@patch("src.alerts.cmms_databricks.databricks_credentials")
@patch("src.utils.databricks_uc.list_uc_catalogs", return_value=["workspace", "samples"])
def test_table_fqn_fixes_missing_main_catalog(mock_catalogs, mock_creds, monkeypatch):
    monkeypatch.setenv("CMMS_DELTA_TABLE", "main.cmapss.cmms_work_orders")
    mock_creds.return_value = ("https://dbc-test.cloud.databricks.com", "pat", "/sql/1.0/warehouses/x")
    with pytest.warns(UserWarning, match="not found"):
        assert table_fqn() == "workspace.cmapss.cmms_work_orders"


def test_create_table_sql_contains_columns():
    sql = create_table_sql("main.cmapss.cmms_work_orders")
    assert "work_order_id" in sql
    assert "USING DELTA" in sql


def test_create_schema_sql():
    assert schema_fqn_from_table("main.cmapss.cmms_work_orders") == "main.cmapss"
    sql = create_schema_sql("main.cmapss.cmms_work_orders")
    assert "CREATE SCHEMA IF NOT EXISTS main.cmapss" in sql


@patch("src.alerts.cmms_databricks.ensure_schema")
@patch("src.alerts.cmms_databricks._connect")
@patch.dict(
    "os.environ",
    {
        "CMMS_LOG_TO_DATABRICKS": "true",
        "DATABRICKS_HOST": "https://dbc-test.cloud.databricks.com",
        "DATABRICKS_TOKEN": "pat-test",
        "DATABRICKS_SQL_WAREHOUSE_ID": "abc123",
        "CMMS_DELTA_TABLE": "main.cmapss.cmms_work_orders",
    },
    clear=False,
)
def test_ensure_table_creates_schema_first(mock_connect, mock_ensure_schema):
    conn = MagicMock()
    cursor = MagicMock()
    mock_connect.return_value.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    from src.alerts.cmms_databricks import ensure_table

    ensure_table(fqn="main.cmapss.cmms_work_orders")
    mock_ensure_schema.assert_called_once_with(fqn="main.cmapss.cmms_work_orders")


@patch.dict(
    "os.environ",
    {
        "CMMS_LOG_TO_DATABRICKS": "false",
    },
    clear=False,
)
def test_insert_skipped_when_disabled(sample_alert):
    out = insert_work_order(
        sample_alert,
        submit_status="mock_logged",
        payload={"asset_id": "ENG-042"},
    )
    assert out["status"] == "skipped"


@patch("src.alerts.cmms_databricks.ensure_table")
@patch("src.alerts.cmms_databricks._connect")
@patch.dict(
    "os.environ",
    {
        "CMMS_LOG_TO_DATABRICKS": "true",
        "DATABRICKS_HOST": "https://dbc-test.cloud.databricks.com",
        "DATABRICKS_TOKEN": "pat-test",
        "DATABRICKS_SQL_WAREHOUSE_ID": "abc123",
        "CMMS_DELTA_TABLE": "main.cmapss.cmms_work_orders",
    },
    clear=False,
)
def test_insert_work_order_calls_sql(mock_connect, _mock_ensure, sample_alert):
    conn = MagicMock()
    cursor = MagicMock()
    mock_connect.return_value.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    out = insert_work_order(
        sample_alert,
        submit_status="mock_logged",
        payload={"asset_id": "ENG-042"},
        dataset_id="FD001",
    )
    assert out["status"] == "databricks_logged"
    assert out["table"] == "main.cmapss.cmms_work_orders"
    assert "explore_url" in out
    assert "dbc-test" in out["explore_url"]
    cursor.execute.assert_called_once()


@patch.dict(
    "os.environ",
    {
        "CMMS_LOG_TO_DATABRICKS": "true",
        "DATABRICKS_HOST": "https://dbc-test.cloud.databricks.com",
        "DATABRICKS_TOKEN": "pat-test",
        "DATABRICKS_SQL_WAREHOUSE_ID": "wh1",
        "CMMS_DELTA_TABLE": "main.cmapss.cmms_work_orders",
    },
    clear=False,
)
def test_is_configured_true():
    assert is_databricks_logging_configured() is True


@patch("src.alerts.cmms_mock.requests.post", side_effect=__import__("requests").RequestException("down"))
@patch("src.alerts.cmms_mock.CMMSClient._maybe_log_databricks")
def test_cmms_client_calls_databricks_hook(mock_db, _post, sample_alert):
    def _merge(_alert, *, submit_status, payload, result, dataset_id):
        return {**result, "databricks": {"status": "ok"}}

    mock_db.side_effect = _merge
    client = CMMSClient()
    result = client.create_work_order(sample_alert, dataset_id="FD002")
    assert result["status"] == "mock_logged"
    mock_db.assert_called_once()
