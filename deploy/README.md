# Live demo — public URL for interviewers

Three-layer stack (Streamlit → FastAPI → Ollama) via Docker Compose, exposed with **one HTTPS tunnel**.

## Prerequisites

1. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** — install, then **start it** and wait until **Engine running** (steady whale icon).  
   Error `dockerDesktopLinuxEngine` / pipe not found → Docker is **not running**; open Docker Desktop first.

2. **cloudflared** (public URL) — correct spelling **`cloudflared`**, not `couldflared`:
   ```powershell
   .\deploy\install_cloudflared.ps1
   ```
   Or: `winget install --id Cloudflare.cloudflared -e`  
   Alternative: `.\deploy\start_tunnel_ngrok.ps1` after `winget install ngrok.ngrok`

3. Demo data imported:

```powershell
cd C:\Users\emana\predictive-maintenance
. .venv\Scripts\activate.ps1
python scripts/import_cmapss_colab_outputs.py --force
```

4. **`.env` at repo root** — required for **CMMS → Databricks** (see below)

Check setup: `.\deploy\preflight.ps1`

## `.env` for CMMS → Databricks (demo)

Work orders are written from the **API container**, which loads repo-root `.env`. Confirm these are set (you already use most of this locally):

```env
CMMS_LOG_TO_DATABRICKS=true
CMMS_DELTA_TABLE=workspace.cmapss.cmms_work_orders
DATABRICKS_HOST=https://dbc-bbbb280e-70f3.cloud.databricks.com
DATABRICKS_TOKEN=<PAT with SQL scope>
DATABRICKS_SQL_HTTP_PATH=/sql/1.0/warehouses/654e1642cb780bc9
```

One-time table setup (on your laptop, same `.env`):

```powershell
python scripts/setup_cmms_databricks_table.py
```

After `.\deploy\run_demo.ps1`, the script prints **CMMS Databricks logging: ON** and the Catalog Explorer URL. In the public app: **Active Alerts** → **Submit work orders** → links to view rows in Databricks.

If logging stays OFF after editing `.env`:

```powershell
docker compose up -d --force-recreate api
```

## One terminal — public URL (recommended)

```powershell
cd C:\Users\emana\predictive-maintenance
git checkout deployment
.\deploy\start_public_demo.ps1
```

This script (1) starts Docker in the background, (2) waits for the API, (3) pulls `llama3.2`, then (4) runs the Cloudflare tunnel **in this same terminal**. Copy the `https://….trycloudflare.com` line when it appears. Press **Ctrl+C** to stop the tunnel (containers keep running until `docker compose down`).

## Start the stack only (no tunnel)

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
| `dockerDesktopLinuxEngine` / pipe not found | **Start Docker Desktop**; wait until Engine running |
| winget `couldflared` not found | Typo — use **cloudflared**; `.\deploy\install_cloudflared.ps1` |

See [docs/deployment.md](../docs/deployment.md) for full checklist.
