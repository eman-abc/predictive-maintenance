# UC5 demo runbook (deployment branch)

One-page checklist for interview / tunnel demos. Replace tunnel host after each `cloudflared` run.

## Prerequisites

| Item | Check |
|------|--------|
| Docker Desktop running | Yes |
| Phase 3 + fleet parquet for **FD003** and **FD004** | `data/processed/cmapss_FD00x_predictions.parquet` |
| `.env` copied from `.env.docker.example` | PAT with **SQL** scope |
| Databricks tables | `python scripts/setup_cmms_databricks_table.py` |
| Cloudflared | `.\deploy\install_cloudflared.ps1` (once) |

## Environment (API container / `.env`)

```env
CMMS_LOG_TO_DATABRICKS=true
CMMS_AUTO_DISPATCH=true
CMMS_DELTA_TABLE=workspace.cmapss.cmms_work_orders
CMMS_AUTO_DELTA_TABLE=workspace.cmapss.cmms_work_orders_auto
DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
DATABRICKS_TOKEN=<pat-with-sql-scope>
DATABRICKS_SQL_HTTP_PATH=/sql/1.0/warehouses/<warehouse-id>
OLLAMA_MODEL=llama3.2
```

Streamlit in Docker uses `API_BASE_URL=http://api:8000` (set in `docker-compose.yml`).

## Start stack

```powershell
cd C:\Users\emana\predictive-maintenance
.\deploy\run_demo.ps1
```

| URL | Purpose |
|-----|---------|
| http://localhost:8080 | Caddy → Streamlit (primary local) |
| http://localhost:8080/api/docs | OpenAPI (show ~30s on slide) |
| http://localhost:8000/health | API health (direct) |

## Public tunnel

```powershell
.\deploy\start_tunnel.ps1
```

Copy the line `https://<random>.trycloudflare.com` from the tunnel terminal (not from `run_demo` output).

Example (changes every run):

- App: `https://jewish-morris-essays-faces.trycloudflare.com`
- API docs: `https://jewish-morris-essays-faces.trycloudflare.com/api/docs`

## Demo flow (5–8 min)

1. **Fleet Overview** — FD003 or FD004 (two fault modes in NASA C-MAPSS).
2. **Asset Detail** — pick a critical engine; **Instant** / **AI briefing** (per-asset).
3. **Active Alerts** — auto-dispatch banner → rows in `cmms_work_orders_auto`; **Shift handover briefing**; **Acknowledge**; manual **Submit to CMMS** → `cmms_work_orders`.
4. **Model Metrics** — FD003 then FD004; Databricks MLflow panel if configured.
5. **Databricks** — Explorer links from Active Alerts expanders.

## Shift briefing vs asset briefing

| | Asset briefing (Asset Detail) | Shift handover (Active Alerts) |
|--|------------------------------|--------------------------------|
| Scope | One `ENG-xxx` | All filtered warning/critical alerts |
| Data | RUL, sensors, health for that row | Alert list summary |
| API | `POST /briefings` | `POST /briefings/shift` |
| Use case | Technician at one engine | Operator handover between shifts |

## Auto-dispatch vs manual CMMS

| Path | Trigger | Delta table | `submit_status` |
|------|---------|-------------|-----------------|
| Auto | Page load (critical) | `cmms_work_orders_auto` | `auto_dispatched` |
| Manual | Button per alert | `cmms_work_orders` | operator submit |

Disable auto: `CMMS_AUTO_DISPATCH=false`.

## API smoke (optional)

```powershell
.\.venv\Scripts\python -m pytest tests/test_api_health.py tests/test_api_cmms.py tests/test_api_shift_ack.py -q
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| 502 on tunnel | Start Docker + `run_demo.ps1` first |
| Auto-dispatch skipped | `CMMS_LOG_TO_DATABRICKS=true` + valid PAT/SQL path |
| Dashboard `ModuleNotFoundError: src` | Rebuild: `docker compose up --build -d` |
| Tunnel URL stale | Restart `start_tunnel.ps1`; update slide link |

## Stop

```powershell
docker compose down
```

Ctrl+C in tunnel terminal stops cloudflared only.
