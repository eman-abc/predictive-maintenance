# Industrial Predictive Maintenance System

End-to-end predictive maintenance for turbofan engines (NASA C-MAPSS): EDA → labeled features → multi-model training → fleet alerts, CMMS routing, and Ollama briefings. Streamlit dashboard locally; FastAPI + Docker + Cloudflare tunnel on the `deployment` branch.

## Features

| Area | Capabilities |
|------|----------------|
| **Data** | C-MAPSS FD001–FD004; optional AI4I 2020 manufacturing dataset |
| **Pipeline** | Phase 1 EDA → Phase 2 labels/preprocess/features → Phase 3 train/score → fleet Parquet |
| **Models** | RUL (RF / GBM / LSTM winner), failure @30 & @72 cycles (GBM), Isolation Forest anomaly, optional Cox PH |
| **MLOps** | MLflow tracking; artifacts in `models/`, `artifacts/`, `mlruns/` |
| **Alerts** | `ThresholdEngine` — health score, RUL/failure/anomaly rules, recommended actions |
| **CMMS** | Mock client locally; Databricks Delta (manual + auto-dispatch for critical) on deployment |
| **Briefings** | Ollama — asset briefing, shift handover, model-metrics Explain (grounded prompts, no RAG) |
| **Dashboard** | Fleet overview, asset detail, active alerts, model metrics |
| **API** | FastAPI (`src/api/`) — fleet, alerts, briefings, CMMS, Phase 3 metrics |

## Documentation

| Doc | Contents |
|-----|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System + data-flow diagrams, modules, deployment stack |
| [docs/cmapss_data_pipeline_diagram.md](docs/cmapss_data_pipeline_diagram.md) | Raw `.txt` → all Parquet files (full pipeline) |
| [docs/cmapss_eda_summary.md](docs/cmapss_eda_summary.md) | Phase 1 EDA findings |
| [docs/cmapss_phase2_preprocessing.md](docs/cmapss_phase2_preprocessing.md) | Preprocessing & feature engineering |
| [docs/cmapss_phase3_modeling.md](docs/cmapss_phase3_modeling.md) | Model comparison, metrics, winner selection |
| [docs/cmapss_alerts_cmms.md](docs/cmapss_alerts_cmms.md) | Alerts, CMMS, P1/P2 SLA routing |
| [docs/cmapss_mlflow_verification.md](docs/cmapss_mlflow_verification.md) | MLflow verification checklist |
| [docs/deployment.md](docs/deployment.md) | Docker Compose, API, env vars |
| [docs/uc5_demo_runbook.md](docs/uc5_demo_runbook.md) | Interview / live demo runbook |
| [deploy/README.md](deploy/README.md) | Quick deploy scripts |

## Project structure

```
predictive-maintenance/
├── configs/           # Per-dataset YAML (from Phase 1 EDA)
├── data/
│   ├── raw/cmapss/    # NASA .txt files
│   └── processed/     # train/test/predictions Parquet
├── artifacts/         # preprocessor.joblib, feature_columns, phase3 summaries
├── models/            # Trained .pkl / .pt
├── src/
│   ├── ingestion/     # Load, EDA, preprocess, features, pipeline
│   ├── models/        # RUL, failure, LSTM, anomaly, survival, Phase 3
│   ├── alerts/        # Threshold engine, payloads, CMMS, Databricks
│   ├── briefings/     # Ollama client & prompts
│   ├── api/           # FastAPI routes (deployment)
│   ├── services/      # Fleet, alerts, briefings, auto-dispatch
│   └── utils/
├── dashboard/         # Streamlit app + api_client (thin client when API_BASE_URL set)
├── scripts/           # download, EDA, build, train, export, deploy helpers
├── deploy/            # run_demo.ps1, tunnel scripts
├── tests/
├── docs/
└── architecture/      # draw.io system diagram
```

## Quick start (local)

### 1. Clone and install

> **Windows:** Use a short path (e.g. `C:\Users\emana\predictive-maintenance`). Enable [Long Paths](https://pip.pypa.io/warnings/enable-long-paths) if `pip install` fails on deep paths.

```bash
cd C:\Users\emana\predictive-maintenance
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
```

### 3. Download datasets

| Dataset | Source | Destination |
|---------|--------|-------------|
| C-MAPSS | [NASA](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) or `python scripts/download_cmapss_data.py` | `data/raw/cmapss/` (`train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`) |
| AI4I 2020 | [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | `data/raw/ai4i/ai4i2020.csv` |

### 4. C-MAPSS pipeline (Phases 1–3)

```bash
python scripts/run_cmapss_eda.py                 # Phase 1 → configs/*.yaml
python scripts/build_cmapss_dataset.py --all   # Phase 2 → train/test Parquet
python scripts/train_cmapss_phase3.py --all    # Phase 3 → models + predictions Parquet
```

One-shot build + train all FD001–FD004:

```bash
python scripts/train_all_cmapss.py
```

**Outputs:** `data/processed/cmapss_*_{train,test,predictions}.parquet`, `artifacts/cmapss_*`, `models/*`, `mlruns/`.

Verify metrics:

```bash
python scripts/report_cmapss_mlflow.py
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

**Faster local train:** `python scripts/train_cmapss_phase3.py --all --skip-lstm --gbm-max-rows 100000`

**Re-export fleet Parquet only:** `python scripts/export_fleet_predictions.py --dataset FD001`

**Colab / GPU:** [notebooks/cmapss_colab_train_all.ipynb](notebooks/cmapss_colab_train_all.ipynb) or `python scripts/cmapss_colab_train.py --fast`

Legacy RF-only path: `python -m src.models.train legacy`

### 5. Launch dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Run tests

```bash
pytest
```

## Live demo (`deployment` branch)

Docker Compose stack: **Streamlit → FastAPI → Ollama**, optional **Databricks** CMMS tables, **Cloudflare** tunnel.

```powershell
git checkout deployment
.\deploy\run_demo.ps1
.\deploy\start_tunnel.ps1
```

Share the `https://….trycloudflare.com` URL (app at `/`, API docs at `/api/docs`). Full checklist: [docs/uc5_demo_runbook.md](docs/uc5_demo_runbook.md).

Set `API_BASE_URL` in the dashboard container so Streamlit calls the FastAPI backend instead of reading Parquet directly.

## Optional: Ollama briefings

Install [Ollama](https://ollama.ai), pull a model, and set in `.env`:

```bash
ollama pull llama3.2
```

```
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
```

## Contributing

Solo workflow: short-lived feature branches, squash merge to `main`, conventional commits. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
