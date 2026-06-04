# Log Colab CMAPSS training to Databricks MLflow

## Prerequisites

1. **Databricks workspace** (Free Edition is fine for UC5 bonus evidence).
2. **Personal Access Token** with scope **`mlflow`** (Settings → Developer → Access tokens).
3. **Experiment path** in the workspace, e.g. `/Shared/predictive_maintenance` (create in UI if missing).

Use only the **base workspace URL** as host, e.g. `https://dbc-xxxx.cloud.databricks.com` — not a `/browse/...` link.

## Colab (recommended)

Open `notebooks/cmapss_colab_train_all.ipynb` on Colab:

1. In **config**, set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` (or Colab secret `DATABRICKS_TOKEN`).
2. Run **§4 Connect MLflow → Databricks** (smoke test logs one run).
3. Run **§5** training — Phase 3 trains **FD001–FD004**; parent runs `FD00X_phase3_summary` with nested `FD00X_rul_cox` (Cox PH) unless you pass `--skip-cox`.
4. In Databricks: **Machine Learning → Experiments** → your experiment path.

Train command (same as the notebook):

```bash
python scripts/cmapss_colab_train.py --fast --upload-dir /content/cmapss_upload --mlflow-databricks
```

## Local smoke test

```bash
python scripts/configure_mlflow_databricks.py \
  --host "https://dbc-xxxx.cloud.databricks.com" \
  --token "$DATABRICKS_TOKEN" \
  --experiment "/Shared/predictive_maintenance" \
  --smoke-test
```

Then train with env vars set (or copy from `.env.example` commented block).

## Verify runs

```bash
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_EXPERIMENT_NAME=/Shared/predictive_maintenance
export DATABRICKS_HOST=...
export DATABRICKS_TOKEN=...
python scripts/report_cmapss_mlflow.py
```

## Model Registry (Databricks → Models)

After Phase 3, registered names (per subset) include:

| Registry name | Role |
|---------------|------|
| `cmapss_rul_{rf,gbm,lstm}_FD00X` | RUL winner |
| `cmapss_failure_30_FD00X` | 30-cycle failure |
| `cmapss_failure_72_FD00X` | 72-cycle failure |
| `cmapss_anomaly_FD00X` | Isolation Forest |
| `cmapss_survival_FD00X` | Cox PH |

Disable with `MLFLOW_REGISTER_MODELS=0`. Re-register from disk:

```bash
python scripts/register_cmapss_models.py --all --mlflow-databricks --run-label uc5_register_v1
```

Colab: notebook section **5b** runs the same script.

## Re-runs (evidence of all trains)

Databricks MLflow **keeps every run**. Re-running the notebook creates new `FD00X_phase3_summary` parents (new `run_id`); nothing is auto-deleted. Use tag **`training_batch`** (default `colab-YYYYMMDDTHHMMSSZ`) or sort by **Start time** to compare sessions. Set `RUN_LABEL` in the notebook config for a memorable tag (e.g. `uc5_colab_full_v1`).

Local files in the Colab VM (`models/`, predictions Parquet) are overwritten each train — download the zip if you need a snapshot.

## Security

- Never commit PATs or put tokens in git.
- Prefer Colab **Secrets** for `DATABRICKS_TOKEN`.
