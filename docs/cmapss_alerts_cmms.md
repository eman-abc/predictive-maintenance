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

- asset_id, alert_id, priority, escalation_tier  
- recommended_action, risk_score, time_to_failure_cycles  
- failure probabilities, anomaly_score, sensor_readings  

On failure → `mock_logged` with payload (demo without live CMMS).

## Refresh alerts after code changes

```bash
python scripts/export_fleet_predictions.py --dataset FD001
python scripts/export_fleet_predictions.py --dataset FD003
```
