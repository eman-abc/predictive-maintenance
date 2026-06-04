# Architecture — Industrial Predictive Maintenance System

## Overview

This system predicts equipment failures and remaining useful life (RUL) from sensor telemetry, generates actionable alerts, and provides operator-facing dashboards and AI briefings.

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Data   │────▶│ Feature Engineer │────▶│  ML Models      │
│ CMAPSS/AI4I │     │ rolling/lag/RUL  │     │ RF/GBM/LSTM/Cox │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                       ┌───────────────────────────────┼───────────────────────┐
                       ▼                               ▼                       ▼
              ┌────────────────┐            ┌──────────────────┐    ┌─────────────────┐
              │ Threshold Engine│            │  MLflow Tracking │    │ Streamlit Dash  │
              │ Alert Generator │            │  Model Registry  │    │ Fleet/Asset/Alerts│
              └───────┬────────┘            └──────────────────┘    └─────────────────┘
                      │
                      ▼
              ┌────────────────┐     ┌──────────────────┐
              │  CMMS (mock)   │     │ Ollama Briefings │
              │  Work Orders   │     │ Shift Summaries  │
              └────────────────┘     └──────────────────┘
```

## Data Flow

1. **Ingestion** — `cmapss_loader.py` and `ai4i_loader.py` parse raw datasets into unified DataFrames.
2. **Feature Engineering** — `FeatureEngineer` computes rolling statistics, lag features, and degradation indices; output saved as Parquet in `data/processed/`.
3. **Training** — `train.py` orchestrates model training with MLflow logging; artifacts saved to `models/`.
4. **Inference** — Models produce RUL estimates and failure probabilities per asset.
5. **Alerting** — `ThresholdEngine` scores risk; `AlertGenerator` builds structured payloads; `CMMSClient` submits work orders.
6. **Briefings** — `OllamaClient` generates natural-language maintenance summaries from model outputs.
7. **Dashboard** — Streamlit pages visualize fleet health, asset trends, alerts, and experiment metrics.

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `src/ingestion/` | Data loading and feature engineering |
| `src/models/` | RUL regression, failure classification, LSTM, survival analysis |
| `src/alerts/` | Risk scoring, alert payloads, CMMS integration |
| `src/briefings/` | LLM client and prompt templates |
| `src/utils/` | Metrics (NASA PHM score) and Plotly/Matplotlib charts |
| `dashboard/` | Streamlit multi-page UI |
| `notebooks/` | Exploratory analysis and threshold tuning |

## Models

| Model | Task | Algorithm | Output |
|-------|------|-----------|--------|
| `rul_regressor.py` | RUL prediction | RF / GBM | Cycles remaining |
| `failure_classifier.py` | Failure detection | RF / GBM | P(failure) |
| `lstm_model.py` | Sequence RUL | PyTorch LSTM | Cycles remaining |
| `survival_model.py` | Time-to-failure | Cox PH (lifelines) | Survival curve |

## Alert Thresholds

Configured via `.env`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `ALERT_RUL_CRITICAL` | 10 | RUL below this triggers critical alert |
| `ALERT_RUL_WARNING` | 30 | RUL below this triggers warning |
| `ALERT_FAILURE_PROB_CRITICAL` | 0.85 | Failure probability critical threshold |
| `ALERT_FAILURE_PROB_WARNING` | 0.60 | Failure probability warning threshold |

Health score (0–100) combines RUL headroom (60%) and inverse failure risk (40%).

## Technology Stack

- **Python 3.10+**
- **scikit-learn** — tabular ML
- **PyTorch** — LSTM sequence models
- **lifelines** — survival analysis (optional)
- **MLflow** — experiment tracking
- **Streamlit + Plotly** — dashboard
- **Ollama** — local LLM briefings
- **Parquet / pandas** — data storage

## Deployment stack (interview / tunnel)

Full runbook: [docs/uc5_demo_runbook.md](docs/uc5_demo_runbook.md). Design notes: [docs/deployment.md](docs/deployment.md).

```
                    ┌─────────────────────────────────────────┐
                    │  Cloudflare quick tunnel (HTTPS)        │
                    └───────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────┐
                    │  Caddy :8080  (reverse proxy)             │
                    │    /        → Streamlit dashboard         │
                    │    /api/*   → FastAPI :8000               │
                    └─────────┬───────────────────┬─────────────┘
                              │                   │
              ┌───────────────▼──────┐   ┌────────▼────────────┐
              │  Streamlit (thin)    │   │  FastAPI backend    │
              │  API_BASE_URL set    │   │  fleet/alerts/CMMS  │
              └──────────────────────┘   │  briefings/shift    │
                                         └─────────┬───────────┘
                                                   │
                         ┌─────────────────────────┼─────────────────────────┐
                         ▼                         ▼                         ▼
                 ┌───────────────┐        ┌──────────────┐        ┌──────────────────┐
                 │ Ollama        │        │ Parquet      │        │ Databricks SQL   │
                 │ briefings     │        │ predictions  │        │ manual + auto    │
                 │               │        │ on disk      │        │ Delta tables     │
                 └───────────────┘        └──────────────┘        └──────────────────┘
```

| Layer | Role |
|-------|------|
| `dashboard/` | UI only when `API_BASE_URL` is set; calls REST |
| `src/api/` + `src/services/` | Business logic, alert ack (in-memory demo), auto-dispatch |
| `src/alerts/cmms_databricks.py` | Manual table + `cmms_work_orders_auto` for critical pipeline |
| `docker-compose.yml` | ollama, api, dashboard, proxy |

Offline training/inference is unchanged; deployment **serves** precomputed Phase 3 parquet.

## Deployment Notes

- Raw datasets and MLflow runs are gitignored; download data locally before training.
- Model artifacts in `models/` are generated by the training pipeline.
- The CMMS client falls back to mock logging when the API is unreachable.
- Ollama runs in Docker on the `deployment` branch; local dev may still use host Ollama.
- Critical alerts auto-write to `CMMS_AUTO_DELTA_TABLE` when `CMMS_AUTO_DISPATCH=true`.

## Diagram

See `architecture/system_diagram.drawio` for an editable draw.io architecture diagram.
