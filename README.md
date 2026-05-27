# Industrial Predictive Maintenance System

An end-to-end predictive maintenance platform for industrial assets, combining RUL regression, failure classification, alert management, LLM-powered briefings, and a Streamlit dashboard.

## Features

- **Data ingestion** — NASA CMAPSS turbofan engine data and AI4I 2020 manufacturing dataset
- **Feature engineering** — rolling statistics, lag features, degradation indices
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
| CMAPSS | [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) | `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` → `data/raw/cmapss/` |
| AI4I 2020 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | `ai4i2020.csv` → `data/raw/ai4i/` |

### 4. Train models

```bash
python -m src.models.train
```

Artifacts are saved to `models/` and metrics logged to `mlruns/`.

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

## License

MIT
