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
3. Run **§5** training — Phase 3 runs appear as `FD00X_phase3_summary`.
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

## Security

- Never commit PATs or put tokens in git.
- Prefer Colab **Secrets** for `DATABRICKS_TOKEN`.
