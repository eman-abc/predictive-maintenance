# Alerts & CMMS (UC5 Component C)

## Thresholds (`ThresholdEngine` + `.env`)

| Signal | Warning | Critical |
|--------|---------|----------|
| Predicted RUL (cycles) | ≤ 30 | ≤ 10 |
| P(failure ≤30 cycles) | ≥ 0.60 | ≥ 0.85 |
| Anomaly score (0–100) | ≥ 55 | ≥ 75 (with RUL ≤30) |

**Classification:** `normal` | `warning` | `critical`  
**Escalation tier:** `L0-Normal` | `L1-Warning` | `L2-Critical` (maps to CMMS priority / SLA)

## UC5 alert fields (predictions parquet)

| Field | Meaning |
|-------|---------|
| `asset_id` | Asset identifier (ENG-001) |
| `sensor_readings_json` | Snapshot of sensor values at last test cycle |
| `risk_score` | Combined health/risk 0–100 |
| `time_to_failure_cycles` | Predicted RUL (cycles until failure) |
| `recommended_action` | Operator-facing maintenance instruction |
| `alert_message` | Full text including model outputs |
| `escalation_tier` | L1/L2 for CMMS routing |

## CMMS integration

`CMMSClient.create_work_order(alert)` sends JSON:

- asset_id, alert_id, escalation_tier, cmms_priority (P1/P2), sla_response_hours, sla_label

**Escalation mapping (implemented):**

| escalation_tier | CMMS priority | SLA |
|-----------------|---------------|-----|
| L2-Critical | P1 | 4h response |
| L1-Warning | P2 | 72h response |
| L0-Normal | P3 | 168h routine |

Also included in the payload: recommended_action, risk_score, time_to_failure_cycles, failure probabilities, anomaly_score, sensor_readings.

On failure → `mock_logged` with payload (demo without live CMMS).

## Databricks work-order audit log

When `CMMS_LOG_TO_DATABRICKS=true`, each button click also **INSERT**s a row into a Unity Catalog **Delta** table (same PAT + SQL warehouse as analytics).

1. Create table once:

```bash
python scripts/setup_cmms_databricks_table.py
```

Or run `scripts/sql/cmms_work_orders.sql` in Databricks SQL (adjust `catalog.schema`).

2. In `.env`:

```env
CMMS_LOG_TO_DATABRICKS=true
CMMS_DELTA_TABLE=workspace.cmapss.cmms_work_orders
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-pat
DATABRICKS_SQL_HTTP_PATH=/sql/1.0/warehouses/654e1642cb780bc9
DATABRICKS_SQL_WAREHOUSE_ID=654e1642cb780bc9
```

**PAT scopes:** the token must include **SQL** (Databricks SQL warehouse). MLflow-only tokens fail with `does not have required scopes: sql`. Generate a new PAT under **User Settings → Developer → Access tokens**, then run:

```bash
python scripts/test_cmms_databricks_connection.py
python scripts/setup_cmms_databricks_table.py
```

3. Submit work orders from **Active Alerts**. After submit, use **Open table in Catalog Explorer** / **Open SQL warehouse** (links returned from the API) or the caption links at the top of the page.

```sql
SELECT * FROM main.cmapss.cmms_work_orders ORDER BY submitted_at DESC;
```

This is an **audit / ops analytics** sink — not a replacement for SAP/Maximo CMMS.

## Refresh alerts after code changes

```bash
python scripts/export_fleet_predictions.py --dataset FD001
python scripts/export_fleet_predictions.py --dataset FD003
```
