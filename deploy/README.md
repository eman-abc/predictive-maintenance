# Live demo — public URL for interviewers

Three-layer stack (Streamlit → FastAPI → Ollama) via Docker Compose, exposed with **one HTTPS tunnel**.

## Prerequisites

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
2. Demo data imported:

```powershell
cd C:\Users\emana\predictive-maintenance
. .venv\Scripts\activate.ps1
python scripts/import_cmapss_colab_outputs.py --force
```

3. `.env` at repo root (copy from `.env.docker.example`, add tokens if using Databricks CMMS log)

## Start the stack

```powershell
.\deploy\run_demo.ps1
```

First run pulls `llama3.2` into the Ollama container (~5 min). Then open:

| URL | Purpose |
|-----|---------|
| http://localhost:8080 | **Public entry** (use for tunnel) |
| http://localhost:8080/api/docs | OpenAPI (architecture demo) |
| http://localhost:8501 | Streamlit direct (debug) |
| http://localhost:8000/health | API direct (debug) |

## Public link (Cloudflare — free, no account for quick tunnel)

```powershell
.\deploy\start_tunnel.ps1
```

Copy the `https://….trycloudflare.com` URL into your slide. Interviewers open:

- App: `https://<tunnel>/`
- API docs: `https://<tunnel>/api/docs`

**Keep your laptop awake** for the duration of the interview.

### ngrok alternative

```powershell
ngrok http 8080
```

## Architecture talking points

- **Thin client:** Streamlit only calls `API_BASE_URL` (no parquet, no Ollama in UI process)
- **Backend:** FastAPI serves fleet/alerts/CMMS; owns Ollama calls
- **AI layer:** Ollama container on Docker network only
- **Lakehouse:** optional Databricks Delta audit log on work-order submit

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Empty fleet | Run `import_cmapss_colab_outputs.py --force` |
| API unhealthy | `docker compose logs api` |
| Ollama slow first time | Wait for `ollama pull` in `run_demo.ps1` |
| Tunnel 502 | `docker compose ps` — all services healthy |

See [docs/deployment.md](../docs/deployment.md) for full checklist.
