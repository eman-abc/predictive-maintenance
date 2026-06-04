# Colab + Databricks — registry troubleshooting

## Your workspace facts

| Setting | Works? |
|---------|--------|
| `MLFLOW_TRACKING_URI=databricks` (Experiments) | **Yes** — you already logged FD001–FD004 |
| Legacy `MLFLOW_REGISTRY_URI=databricks` | **No** — `PERMISSION_DENIED: legacy workspace model registry is disabled` |
| Unity Catalog Model Registry | **Only** path for “Models” UI — needs catalog + schema + PAT permissions |
| `main` catalog | **No** on your workspace — `CATALOG_DOES_NOT_EXIST` |
| List catalogs API (`list_uc_catalogs.py`) | **403** — PAT lacks Unity Catalog API scope |

**Do not** follow advice to use `mlflow.set_registry_uri("databricks")` or `MLFLOW_ENABLE_UNITY_CATALOG=false` — legacy is off.

## For the interview (minimum)

You already have:

- All four `FD00X_phase3_summary` runs in **Experiments**
- Nested RF / GBM / LSTM / Cox / failure / anomaly runs
- Local zip with `models/` (21 files) for Streamlit on your PC

That satisfies **MLOps / experiment tracking**. Unity Catalog **Models** is optional.

## Missing `artifacts/cmapss_FD00X_feature_columns.json`

The zip often has `models/` and `data/processed/` but not every `artifacts/*.json`. After `git pull`, registration uses **feature columns stored inside the `.pkl` files** or inferred from `data/processed/cmapss_FD00X_train.parquet`.

Quick check:

```python
from pathlib import Path
print("models", len(list(Path("models").glob("*"))))
print("train parquet", Path("data/processed/cmapss_FD001_train.parquet").exists())
```

## Option A — Skip registry (recommended now)

```python
import os
os.environ["MLFLOW_REGISTER_MODELS"] = "log_only"
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = token  # use Colab secret, not pasted in notebook
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT

!python scripts/register_cmapss_models.py --all --mlflow-databricks --run-label colab-artifacts-only
```

Models appear under each `FD00X_registry_only` run → **Artifacts**, not **Catalog → Models**.

## Option B — Unity Catalog registry (needs admin / new PAT)

1. Databricks → **Catalog Explorer** → write down **catalog** and **schema** (not `main` unless you see it).
2. Create PAT with **Unity Catalog** permissions (or ask admin for `USE CATALOG`, `USE SCHEMA`, `CREATE MODEL`).
3. Colab after `git pull` + `pip install -r requirements.txt`:

```python
import os
os.environ["MLFLOW_REGISTER_MODELS"] = "1"
os.environ.pop("MLFLOW_REGISTRY_URI", None)  # do NOT set to "databricks"
os.environ["MLFLOW_UC_CATALOG"] = "YOUR_CATALOG_FROM_UI"
os.environ["MLFLOW_UC_SCHEMA"] = "YOUR_SCHEMA_FROM_UI"
# ... DATABRICKS_HOST, TOKEN, TRACKING_URI, EXPERIMENT ...

!python scripts/register_cmapss_models.py --dataset FD001 --mlflow-databricks
```

## Colab session hygiene

1. **Runtime → Run all** from top after disconnect: **§2 pip install** before any `import mlflow`.
2. **`git pull origin main`** — fixes `registry/` slash error (need code ≥ `5d58db6`).
3. **Do not** put comments on `%cd` lines.
4. **Revoke and recreate PAT** if you pasted it in a notebook or chat.

## Zip on PC (Windows)

```powershell
cd C:\Users\emana\predictive-maintenance
Expand-Archive -Path .\cmapss_colab_outputs.zip -DestinationPath . -Force
streamlit run dashboard/app.py --server.port 8502
```

Use forward slashes in Python; `dir` not `dir \` in PowerShell for checks.
