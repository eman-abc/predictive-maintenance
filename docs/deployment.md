# Deployment Guide — UC5 Live Demo

This document takes the project from **local development** to a **public interview URL** while preserving a clear **three-layer architecture**:

| Layer | Role | Technology |
|-------|------|------------|
| **Frontend (thin client)** | UI only — charts, tables, buttons | Streamlit + Plotly |
| **Backend (API)** | Business logic, data access, alert rules, CMMS | FastAPI |
| **AI layer** | LLM inference (isolated) | Ollama (`llama3.2`) |

**Inference / training** stays **offline** (batch Phase 3 → predictions parquet). The deployed stack **serves** precomputed scores and calls the AI layer only for briefings and metric explanations.

---

## Does a tunneled local stack “count” as proper architecture?

**Yes — if you deploy the three layers as separate services**, not as a single fat Streamlit process.

| Setup | Public URL? | Architecture story |
|-------|-------------|-------------------|
| Tunnel → monolithic Streamlit (reads parquet + calls Ollama in-process) | Yes | **Weak** — looks like one app; hard to defend “thin client + API + AI” |
| **Docker Compose** (Ollama + FastAPI + Streamlit) + tunnel to one entrypoint | Yes | **Strong** — same diagram as production; tunnel is only transport |
| Streamlit on Render + API on Render + Ollama on laptop via tunnel | Yes | **Strong** — “private LLM endpoint”; more moving parts |

**Recommended for this repo:** Compose on your laptop (8 GB RAM for Ollama) + **one HTTPS tunnel** to a small **reverse proxy** (or Streamlit with `API_BASE_URL` pointing at internal `http://api:8000`).

Interviewers see **one URL**. You show:

- **Fleet UI** — frontend
- **`https://<demo>/api/docs`** — backend contract (optional but impressive)
- **Architecture slide** — three boxes + “batch training offline”

Latency through ngrok/Cloudflare is usually **fine for a demo** (tens of ms extra; Ollama dominates briefing time).

---

## Target runtime architecture

```
                    Internet (interviewers)
                              │
                    Cloudflare Tunnel / ngrok
                              │
                    ┌─────────▼─────────┐
                    │  Reverse proxy     │  optional: Caddy/nginx
                    │  :443 → routes     │  / → Streamlit :8501
                    └─────────┬─────────┘  /api → FastAPI :8000
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  dashboard   │    │     api      │    │    ollama    │
  │  Streamlit   │───▶│   FastAPI    │───▶│  :11434      │
  │  thin client │    │   :8000      │    │  llama3.2    │
  └──────────────┘    └──────┬───────┘    └──────────────┘
                             │
                    reads (volume mount)
                    data/processed/*.parquet
                    artifacts/*.json
```

**Rules**

- Streamlit **never** reads parquet or calls Ollama directly.
- FastAPI is the **only** service that talks to Ollama (`OLLAMA_BASE_URL=http://ollama:11434`).
- No model training or batch scoring in the request path.

---

## Prerequisites

| Item | Notes |
|------|--------|
| Docker Desktop | Windows/macOS/Linux |
| Ollama (host or container) | `ollama pull llama3.2` |
| Colab outputs imported | See Phase 0 |
| `.env` | Copy from `.env.example` |
| ~8 GB free RAM | For Ollama + API + Streamlit |
| Cloudflare Tunnel or ngrok | For public HTTPS URL |

**Not required for demo:** Render, Kubernetes, live CMMS, Unity Catalog registration.

---

## Phase 0 — Demo data on disk

The dashboard and API expect artifacts at the **repo root**, not only under `cmapss_colab_outputs/`:

```powershell
cd C:\Users\emana\predictive-maintenance
. .venv\Scripts\activate.ps1
python scripts/import_cmapss_colab_outputs.py --force
```

Verify:

```powershell
dir data\processed\cmapss_FD001_predictions.parquet
dir artifacts\cmapss_FD001_phase3_summary.json
dir models\rul_gbm_FD001.pkl
```

**Interview minimum:** FD001–FD004 all imported (~500 MB).  
**Render free fallback:** FD001 only if you later deploy a slim cloud UI.

---

## Phase 1 — FastAPI backend (new)

### 1.1 Dependencies

Add to `requirements.txt` (or `requirements-api.txt`):

```text
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
```

### 1.2 Suggested layout

```text
src/api/
  __init__.py
  main.py              # FastAPI app, CORS, lifespan (Ollama warmup)
  deps.py              # shared paths, dataset_id validation
  routes/
    health.py
    fleet.py
    assets.py
    alerts.py
    metrics.py
    briefings.py
    cmms.py
src/services/
  fleet_service.py     # move logic from dashboard/data_loader.py
  alerts_service.py    # ThresholdEngine + payload build
  metrics_service.py   # phase3 summaries, registry
  briefing_service.py  # wraps OllamaClient + metric_explanation_prompts
```

### 1.3 API contract (v1)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + Ollama reachable flag |
| `GET` | `/datasets` | List FD subsets with predictions |
| `GET` | `/fleet?dataset=FD001` | Fleet table rows |
| `GET` | `/assets/{asset_id}?dataset=FD001` | Asset detail + sensor series |
| `GET` | `/alerts?dataset=FD001` | Active alerts |
| `GET` | `/metrics/phase3?dataset=FD001` | Phase 3 summary JSON |
| `GET` | `/metrics/registry` | Training registry |
| `GET` | `/mlflow/links?dataset=FD001` | Databricks run URLs (if configured) |
| `POST` | `/briefings` | Body: asset context → briefing text |
| `POST` | `/metrics/explain` | Body: `section`, `dataset` → explanation |
| `POST` | `/cmms/workorders` | Alert payload → mock CMMS response |

Reuse existing modules:

- `src/alerts/threshold_engine.py`, `alert_payload.py`, `cmms_mock.py`
- `src/briefings/ollama_client.py`, `metric_explanation_prompts.py`
- `dashboard/mlflow_links.py` (or move to `src/services/mlflow_links.py`)

### 1.4 Run locally (before Docker)

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` and smoke-test `/health`, `/fleet?dataset=FD001`.

### 1.5 Tests

```powershell
pytest tests/test_api_health.py tests/test_api_fleet.py
```

Add minimal API tests with `TestClient` and fixture parquet under `tests/fixtures/` (or skip if FD001 data missing).

---

## Phase 2 — Thin Streamlit client

### 2.1 Environment

```env
API_BASE_URL=http://localhost:8000
```

In Docker Compose (internal network):

```env
API_BASE_URL=http://api:8000
```

### 2.2 Refactor rule

Replace direct use of `dashboard/data_loader.py` parquet reads with an HTTP client module, e.g. `dashboard/api_client.py`:

- `get_datasets()`, `get_fleet(dataset)`, `get_asset(...)`, `get_alerts(...)`, etc.
- Keep **Plotly** and layout code in pages; only data fetching changes.

Pages to update:

- `dashboard/app.py`
- `dashboard/pages/01_fleet_overview.py`
- `dashboard/pages/02_asset_detail.py`
- `dashboard/pages/03_active_alerts.py`
- `dashboard/pages/04_model_metrics.py`

Briefings and metric **Explain** buttons call `POST` on the API (not `get_ollama_client()` in Streamlit).

### 2.3 Run locally (two terminals)

```powershell
# Terminal 1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2
$env:API_BASE_URL = "http://localhost:8000"
streamlit run dashboard/app.py --server.port 8501
```

### 2.4 Acceptance

- [ ] Sidebar dataset switch works for all imported FD subsets
- [ ] Asset briefing and Model Metrics Explain hit API (network tab or API logs)
- [ ] Ollama stopped → instant/template fallback still returns text
- [ ] Streamlit container/process has **no** `OLLAMA_BASE_URL` required

---

## Phase 3 — Docker Compose

### 3.1 Files to add

```text
docker/
  api/Dockerfile
  dashboard/Dockerfile
docker-compose.yml
docker-compose.demo.yml   # optional: FD001-only slim image
.env.docker.example
```

### 3.2 Example `docker-compose.yml`

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"   # optional: debug from host only
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 5

  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
    env_file: .env
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    volumes:
      - ./data/processed:/app/data/processed:ro
      - ./artifacts:/app/artifacts:ro
    depends_on:
      ollama:
        condition: service_healthy
    ports:
      - "8000:8000"

  dashboard:
    build:
      context: .
      dockerfile: docker/dashboard/Dockerfile
    environment:
      API_BASE_URL: http://api:8000
    depends_on:
      - api
    ports:
      - "8501:8501"

volumes:
  ollama_data:
```

### 3.3 First-time Ollama model pull

After `docker compose up -d ollama`:

```powershell
docker compose exec ollama ollama pull llama3.2
```

Or bake into a custom `docker/ollama/Dockerfile` with `ollama pull` at build time (larger image, faster first demo).

### 3.4 API image notes

- Use `requirements-api.txt` without `torch` if API only serves parquet (smaller image).
- Working directory `/app`, `PYTHONPATH=/app`.
- CMD: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`

### 3.5 Dashboard image notes

- Lighter deps: `streamlit`, `plotly`, `httpx`, `pandas` (for display only if needed).
- CMD: `streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0`

### 3.6 Run stack

```powershell
docker compose up --build
```

Verify:

- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

---

## Phase 4 — Single public URL (tunnel)

Expose **one HTTPS URL** to interviewers. Two patterns:

### Option A — Tunnel Streamlit only (simplest)

- Tunnel port **8501** → public URL.
- Streamlit uses `API_BASE_URL=http://api:8000` on the **Docker internal network** (works inside Compose).
- For panel: share Streamlit URL; optionally share API docs via second tunnel on **8000** or “localhost only” during screen share.

**Cloudflare Tunnel (quick):**

```powershell
cloudflared tunnel --url http://localhost:8501
```

**ngrok:**

```powershell
ngrok http 8501
```

### Option B — Reverse proxy + one tunnel (best architecture demo)

Add a **Caddy** or **nginx** service in Compose:

| Path | Backend |
|------|---------|
| `/` | `dashboard:8501` |
| `/api` | `api:8000` |

Tunnel port **443/80** on the proxy. Interviewers get:

- App: `https://<random>.trycloudflare.com/`
- API docs: `https://<random>.trycloudflare.com/api/docs`

Set `API_BASE_URL` to the **public** `/api` prefix for browser-side calls, or keep internal URL for server-side Streamlit→API (preferred: Streamlit server-side calls `http://api:8000`, browser only talks to Streamlit).

### Tunnel checklist (day of interview)

- [ ] Laptop plugged in, sleep disabled
- [ ] `docker compose up` healthy
- [ ] `ollama list` shows `llama3.2`
- [ ] Open public URL on **phone** (not same machine) — test fleet + one briefing
- [ ] Copy URL into slide; keep `http://localhost:8501` as backup
- [ ] If tunnel dies: restart tunnel; fallback to screen share

### Is the tunnel slow?

Usually **acceptable** (~50–200 ms extra). Ollama generation (seconds) dominates. Render free cold starts are often **worse** than a warm local stack through a tunnel.

---

## Phase 5 — Optional cloud frontend (Render)

Use only if you want a **second** always-on URL without your laptop.

| Service | Render plan | Role |
|---------|-------------|------|
| Dashboard | Free (512 MB) | Thin client; `API_BASE_URL` → your laptop tunnel **or** cloud API |
| API | Free / Starter | Not viable with Ollama on same 512 MB instance |

**Realistic Render setup:** API + Streamlit on Render, `OLLAMA_BASE_URL` → Cloudflare tunnel to **laptop Ollama** during interview (laptop must stay on). Otherwise use **instant** briefings on Render only.

**Not recommended:** Full FD001–FD004 parquet in Render free image (size + RAM).

---

## Environment variables

| Variable | Used by | Example (Compose) |
|----------|---------|-------------------|
| `API_BASE_URL` | Streamlit | `http://api:8000` |
| `OLLAMA_BASE_URL` | FastAPI only | `http://ollama:11434` |
| `OLLAMA_MODEL` | FastAPI | `llama3.2` |
| `OLLAMA_NUM_PREDICT_METRICS` | FastAPI | `320` |
| `DATABRICKS_HOST` | API (optional) | From `.env` |
| `DATABRICKS_TOKEN` | API (optional) | From `.env` |
| `MLFLOW_EXPERIMENT_ID` | API / metrics | Registry fallback |
| `ALERT_RUL_WARNING` etc. | API alerts | From `.env.example` |
| `CMMS_API_URL` | API | Mock default |

Never commit `.env` with tokens.

---

## Git branches

| Branch | Purpose |
|--------|---------|
| `main` | Feature development |
| `deployment` | `docker-compose.yml`, Dockerfiles, `src/api/`, thin client, `docs/deployment.md` |

Merge `main` → `deployment` when Phases 1–4 pass local acceptance.

---

## Completion checklist

### Implementation

- [ ] Phase 0: `import_cmapss_colab_outputs.py --force` (you run once on laptop)
- [x] Phase 1: FastAPI (`src/api/main.py`, `src/services/`) + `tests/test_api_health.py`
- [x] Phase 2: Thin Streamlit when `API_BASE_URL` is set (`dashboard/api_client.py`)
- [x] Phase 3: `docker-compose.yml` + `deploy/run_demo.ps1`
- [ ] Phase 4: Public tunnel URL tested from phone (`deploy/start_tunnel.ps1`)

### Architecture (presentation)

- [ ] Slide: three layers + offline batch training
- [ ] Live: open `/api/docs` or describe internal `http://api:8000`
- [ ] Live: one Ollama briefing + one metric Explain
- [ ] Mention: CMMS mock, Databricks MLflow links, instant LLM fallback

### Security (talking points)

- Ollama not exposed publicly (only API on Docker network; tunnel terminates at Streamlit/proxy)
- Secrets in `.env`, not in git
- CMAPSS has no PII

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Empty fleet | Run import script; check `data/processed` mount in API container |
| `OllamaClient.generate() got unexpected keyword argument` | Restart API; use fresh client (see `dashboard/ollama_startup.py` reload pattern in API) |
| Briefing timeout | Lower `OLLAMA_NUM_PREDICT`; use instant fallback |
| Explain cut off | Raise `OLLAMA_NUM_PREDICT_METRICS` in `.env` |
| Tunnel 502 | Compose not up; wrong port; Windows firewall |
| Cox RMSE shows `—` | Expected on FD001; concordance is the Cox signal |

---

## Implementation order (summary)

1. **Phase 0** — data import (30 min)  
2. **Phase 1** — FastAPI (2–3 days)  
3. **Phase 2** — thin Streamlit (1–2 days)  
4. **Phase 3** — Docker Compose (1 day)  
5. **Phase 4** — tunnel + interview dry run (half day)  
6. **Presentation** — after deployment is stable  

---

## Related docs

- [ARCHITECTURE.md](../ARCHITECTURE.md) — module overview  
- [cmapss_phase3_modeling.md](cmapss_phase3_modeling.md) — metrics and UC5 mapping  
- [cmapss_alerts_cmms.md](cmapss_alerts_cmms.md) — alert fields  
- [mlflow_databricks_colab.md](mlflow_databricks_colab.md) — experiment evidence  
- [UC5 requirement PDF](project_requirement/UC5_Eman_Chaudhary.pdf) — assignment spec  

---

*Last updated: deployment architecture targets Docker Compose on the developer laptop with HTTPS tunnel for the interview URL. Implement Phases 1–2 in code before relying on the tunnel for the panel demo.*
