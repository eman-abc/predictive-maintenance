-- Run in Databricks SQL (replace catalog/schema if needed).
-- Matches src/alerts/cmms_databricks.py schema.

CREATE SCHEMA IF NOT EXISTS main.cmapss
COMMENT 'UC5 predictive maintenance — CMMS work order audit schema';

CREATE TABLE IF NOT EXISTS main.cmapss.cmms_work_orders (
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
COMMENT 'UC5 CMMS work order audit log';
