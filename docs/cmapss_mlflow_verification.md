# CMAPSS training — MLflow verification for supervisors

Phase 3 logs one **parent MLflow run per dataset** (`FD00X_phase3_summary`) with nested runs for each model. All parent runs are tagged `pipeline=cmapss_phase3` for filtering.

## Re-runs and evidence

**MLflow (local `./mlruns` or Databricks) never replaces old runs.** Each train call creates a new run with a new `run_id`. You can have many `FD001_phase3_summary` rows in the UI — filter or sort by **Start time** or tag **`training_batch`** (e.g. `colab-20260602T120000Z`).

**Local disk is different:** `models/`, `data/processed/*_predictions.parquet`, and the latest row in `artifacts/cmapss_training_registry.json` reflect only the **most recent** train. Prior MLflow runs remain your audit trail; `mlflow_run_history` in the registry keeps the last 20 local run IDs per subset.

## 1. Train all four NASA subsets

```bash
# Phase 2 build + Phase 3 train (long on FD002/FD004)
python scripts/train_all_cmapss.py

# Or if Parquet already exists:
python scripts/train_all_cmapss.py --skip-build

# Phase 3 only:
python scripts/train_cmapss_phase3.py --all
```

### Fast training (still fully logged in MLflow)

```bash
python scripts/train_cmapss_phase3.py --all --skip-lstm --gbm-max-rows 100000 --anomaly-max-rows 50000
python scripts/train_all_cmapss.py --skip-build --skip-lstm --gbm-max-rows 100000
python -m src.models.train fast
```

Parameters `skip_lstm`, `skip_cox`, `gbm_max_train_rows`, `cox_max_train_rows`, and `anomaly_max_train_rows` appear on each `FD00X_phase3_summary` run. Cox test metrics appear as `test_cox_rmse`, `test_cox_rul_score` on the parent run.

### Google Colab

Open **`notebooks/cmapss_colab_train_all.ipynb`** (Runtime → GPU optional), or run:

```bash
python scripts/cmapss_colab_train.py --fast
```

Defaults: `python -m src.models.train` runs **FD001–FD004** (full). Use `python -m src.models.train fast` for the quick path.

## 2. Verification report (terminal)

```bash
python scripts/report_cmapss_mlflow.py
```

Exit code **0** when every subset has:

- `data/processed/cmapss_FD00X_predictions.parquet`
- An MLflow run named `FD00X_phase3_summary` with tag `pipeline=cmapss_phase3`

JSON output for slides or CI:

```bash
python scripts/report_cmapss_mlflow.py --json
```

## 3. MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

In the UI:

1. Open experiment **`predictive_maintenance`** (or your `MLFLOW_EXPERIMENT_NAME`).
2. Filter: `tags.pipeline = "cmapss_phase3"`.
3. Open each **`FD00X_phase3_summary`** run:
   - **Parameters:** `dataset_id`, `winner`, `n_features`, `train_units`
   - **Metrics:** `test_rmse`, `test_rul_score`, `test_cox_rmse`, `test_cox_rul_score`, `winner_val_*`
   - **Artifacts:** `phase3_summary.json` (full JSON), `models/survival_FD00X.pkl` (Cox PH)

Nested runs under each summary: `FD00X_rul_rf`, `FD00X_rul_gbm`, `FD00X_rul_lstm`, `FD00X_rul_cox`, `FD00X_failure_30_gbm`, `FD00X_failure_72_gbm`, `FD00X_anomaly_iforest`.

## 4. Local index (no MLflow UI)

| File | Purpose |
|------|---------|
| `artifacts/cmapss_training_registry.json` | Merged index of all trained subsets + MLflow run IDs |
| `artifacts/cmapss_FD00X_phase3_summary.json` | Per-dataset metrics snapshot |
| `data/processed/cmapss_FD00X_predictions.parquet` | Fleet export for Streamlit |

## 5. Feature engineering (canonical path)

CMAPSS features are **not** produced by legacy `FeatureEngineer.engineer_cmapss()`. They come from Phase 2:

```bash
python scripts/build_cmapss_dataset.py --all
```

See [cmapss_phase2_preprocessing.md](cmapss_phase2_preprocessing.md).

## 6. Dashboard

After all four subsets are trained:

```bash
streamlit run dashboard/app.py
```

The sidebar lists every FD subset that has a predictions Parquet file.
