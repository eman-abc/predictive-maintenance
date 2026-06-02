# Industrial Predictive Maintenance System

An end-to-end predictive maintenance platform for industrial assets, combining RUL regression, failure classification, alert management, LLM-powered briefings, and a Streamlit dashboard.

## Features

- **Data ingestion** — NASA CMAPSS turbofan engine data and AI4I 2020 manufacturing dataset
- **Feature engineering** — Phase 2 CMAPSS pipeline (rolling, lag, delta, slope, spectral, degradation index); config per FD001–FD004
- **ML models** — Random Forest / GBM for RUL and failure prediction, LSTM for sequences, optional Cox PH survival analysis
- **MLOps** — MLflow experiment tracking and model artifact storage
- **Alerts** — configurable threshold engine with CMMS work order integration (mock)
- **AI briefings** — Ollama-powered maintenance summaries for operators
- **Dashboard** — Streamlit multi-page app for fleet monitoring

## Project Structure

```
predictive-maintenance/
├── data/           # Raw, processed, and synthetic datasets
├── src/            # Core Python packages
├── dashboard/      # Streamlit application
├── notebooks/      # Exploratory analysis
├── models/         # Trained model artifacts
├── tests/          # Unit tests
└── architecture/   # System diagrams
```

## Quick Start

### 1. Clone and install

> **Windows note:** Keep the project in a short path (e.g. `C:\Users\emana\predictive-maintenance`).
> Deep paths under `.cursor\projects\...` can exceed Windows' 260-character limit and cause `pip install` to fail.
> Enable [Long Paths](https://pip.pypa.io/warnings/enable-long-paths) or use a shorter directory.

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

Place files in the following directories:

| Dataset | Source | Files |
|---------|--------|-------|
| CMAPSS | [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) or `python scripts/download_cmapss_data.py` | `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt` (X=001–004) → `data/raw/cmapss/` |
| AI4I 2020 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | `ai4i2020.csv` → `data/raw/ai4i/` |

### 4. Build features and train (CMAPSS)

```bash
python scripts/build_cmapss_dataset.py --all
python scripts/train_cmapss_phase3.py --all
```

Or one command (build + train all FD001–FD004):

```bash
python scripts/train_all_cmapss.py
```

Verify for demos / supervisor review:

```bash
python scripts/report_cmapss_mlflow.py
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

See [docs/cmapss_mlflow_verification.md](docs/cmapss_mlflow_verification.md).

**Colab / cloud GPU:** [notebooks/cmapss_colab_train_all.ipynb](notebooks/cmapss_colab_train_all.ipynb) or `python scripts/cmapss_colab_train.py --fast`

**Faster local train (MLflow unchanged):** `python scripts/train_cmapss_phase3.py --all --skip-lstm --gbm-max-rows 100000`

Artifacts: `models/`, `data/processed/cmapss_*_predictions.parquet`, `artifacts/cmapss_*_phase3_summary.json`, metrics in `mlruns/`.

Legacy quick train (RF on Parquet, no full Phase 3 comparison): `python -m src.models.train legacy`

### 5. Launch dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Run tests

```bash
pytest
```

## Optional: Ollama Briefings

Install [Ollama](https://ollama.ai) and pull a model:

```bash
ollama pull llama3.2
```

Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env`.

## Contributing

Solo workflow: short-lived feature branches, squash merge to `main`, conventional commits. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
