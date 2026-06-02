# CMAPSS Phase 3 — Modeling, Evaluation & Operations

Phase 3 trains and compares RUL models, selects a winner by **NASA score**, scores the official **test last-cycle** benchmark, and feeds **alerts + Streamlit** from real predictions.

**Prerequisites:** Phase 2 Parquet for your dataset (e.g. `cmapss_FD001_train/test.parquet`).

---

## Locked decisions

| Topic | Choice |
|-------|--------|
| Winner selection | **Lowest validation NASA score** (`rul_score`); tie-break **RMSE** then simpler model (RF < GBM < LSTM) |
| Test evaluation | **Last cycle per engine** only |
| LSTM window | **30** cycles |
| Failure probability | **`failure_30` / `failure_72` GBM** (UC5 24h/72h → 30/72 cycles); alerts use `failure_30` |
| Anomaly / degradation | **Isolation Forest** on healthy train rows (RUL≥30); `anomaly_score` 0–100 on fleet export |
| Failure clf validation | Non-terminal cycles on val engines (both classes) |
| Failure clf test metrics | Last cycle per test engine (operational snapshot) |
| Train/val split | **80/20 by `unit_id`** on train engines; test never used for selection |
| Validation scoring | **All non-terminal cycles** on val engines (exclude EOL row where RUL=0) |
| Test evaluation | **Last cycle per engine** only (official benchmark; test RUL is not all zero) |
| Final models | Retrained on **all train engines** before test scoring (GBM subsamples if rows > 250k) |

---

## Run

```bash
python scripts/train_cmapss_phase3.py --all          # FD001–FD004 (default)
python scripts/train_cmapss_phase3.py --dataset FD001
python scripts/train_all_cmapss.py                   # build + train all
python scripts/run_e2e.py --skip-build
python scripts/report_cmapss_mlflow.py               # supervisor verification
```

See [cmapss_mlflow_verification.md](cmapss_mlflow_verification.md).

Options: `--lstm-epochs 15`, `--val-fraction 0.2`

---

## Outputs

| Artifact | Path |
|----------|------|
| Best RUL model | `models/rul_{rf,gbm}_FD001.pkl` or `models/rul_lstm_FD001.pt` |
| Failure classifier | `models/failure_30_FD001.pkl` |
| Fleet predictions | `data/processed/cmapss_FD001_predictions.parquet` (columns: `failure_prob_30`, `failure_prob_72`, `dataset_id`; `failure_prob` = 30-cycle prob for alerts) |
| Summary JSON | `artifacts/cmapss_FD001_phase3_summary.json` |
| MLflow runs | `./mlruns` experiment `predictive_maintenance` |

---

## Metrics

- **RMSE** — root mean squared error in cycles (interpretable).
- **NASA score** — asymmetric PHM08 metric; under-predicting RUL is penalized more (safer for maintenance narrative).

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Use the sidebar **CMAPSS dataset** selector (FD001, FD003, …) on all dashboard pages.

---

## UC5 alignment

| Component | Phase 3 |
|-----------|---------|
| B — Model comparison | RF, GBM, LSTM + MLflow |
| B — Metrics | NASA score + RMSE on test |
| C — Alerts | ThresholdEngine + failure_30 probability |
| D — Dashboard | Fleet, asset, alerts, MLflow page |

---

## References

- [Phase 2 preprocessing](cmapss_phase2_preprocessing.md)
- `src/models/cmapss_phase3.py`
