# CMAPSS data pipeline — raw files to Parquet (complete diagram)

End-to-end path from NASA `.txt` files through EDA, labels, preprocessing, feature engineering, training, and fleet Parquet used by the dashboard.

**Commands (in order):**

```bash
python scripts/download_cmapss_data.py          # optional
python scripts/run_cmapss_eda.py                # Phase 1
python scripts/build_cmapss_dataset.py --all    # Phase 2
python scripts/train_cmapss_phase3.py --all     # Phase 3 (+ predictions parquet)
```

---

## Master flow (all phases → all Parquet outputs)

```mermaid
flowchart TB
  subgraph RAW["0. Raw NASA C-MAPSS"]
    T["train_FD00X.txt"]
    TE["test_FD00X.txt"]
    R["RUL_FD00X.txt"]
  end

  subgraph P1["Phase 1 — EDA (no Parquet yet)"]
    EDA["scripts/run_cmapss_eda.py\nsrc/ingestion/cmapss_eda.py"]
    CFG["configs/cmapss_FD001.yaml … FD004.yaml"]
    DOC["docs/cmapss_eda_summary.md"]
    SNAP["configs/cmapss_eda_snapshot.json"]
  end

  subgraph P2["Phase 2 — Labels + preprocess + features"]
    BUILD["scripts/build_cmapss_dataset.py\nsrc/ingestion/cmapss_pipeline.py"]
    PQ_TR["data/processed/cmapss_FD00X_train.parquet"]
    PQ_TE["data/processed/cmapss_FD00X_test.parquet"]
    PREP["artifacts/cmapss_FD00X_preprocessor.joblib"]
    FEAT["artifacts/cmapss_FD00X_feature_columns.json"]
  end

  subgraph P3["Phase 3 — Train + score"]
    TRAIN["scripts/train_cmapss_phase3.py\nsrc/models/cmapss_phase3.py"]
    MOD["models/*.pkl, *.pt"]
    SUM["artifacts/cmapss_FD00X_phase3_summary.json"]
    REG["artifacts/cmapss_training_registry.json"]
    MLF["mlruns/ or Databricks MLflow"]
    PQ_PR["data/processed/cmapss_FD00X_predictions.parquet"]
  end

  subgraph UI["Dashboard / API (read-only)"]
    DASH["Streamlit + FastAPI\nload parquet / JSON"]
  end

  T & TE & R --> EDA
  EDA --> CFG & DOC & SNAP
  CFG --> BUILD
  T & TE & R --> BUILD
  BUILD --> PQ_TR & PQ_TE & PREP & FEAT
  PQ_TR & PQ_TE --> TRAIN
  PREP & FEAT --> TRAIN
  TRAIN --> MOD & SUM & REG & MLF & PQ_PR
  PQ_PR --> DASH
  PQ_TE --> DASH
  SUM --> DASH
```

---

## Phase 1 — EDA detail (feeds YAML, not Parquet)

```mermaid
flowchart LR
  subgraph IN["Inputs"]
    RAW2["data/raw/cmapss/\ntrain, test, RUL"]
  end

  subgraph ANALYZE["cmapss_eda.analyze_dataset()"]
    U["Fleet stats\nengines, cycle lengths"]
    S["Sensor variance QC\nconstant / near-constant"]
    O["Operating conditions\nop triplets, 6 scenarios?"]
    T2["Test censorship\nRUL at last cycle\nimplied failure cycle"]
  end

  subgraph OUT["Outputs"]
    YML["configs/cmapss_FD00X.yaml\n• sensors.drop / keep\n• cluster_for_normalization\n• rolling_windows, lags\n• spectral_sensors\n• eda_snapshot stats"]
    MD["docs/cmapss_eda_summary.md"]
    JS["configs/cmapss_eda_snapshot.json"]
  end

  RAW2 --> U & S & O & T2
  U & S & O & T2 --> YML & MD & JS
```

| EDA decision | Example FD001 | Used in Phase 2 |
|--------------|---------------|-----------------|
| Drop constant sensors | 7 sensors dropped | `CmapssPreprocessor` |
| Keep sensors | 14 kept | Feature columns |
| Op clustering | `false` (1 condition) | Skip KMeans |
| Op clustering | `true` (FD002/004) | KMeans + cluster scaler |
| Spectral sensors | First 5 kept sensors | `CmapssFeatureEngineer` |
| Train filter RUL | `< 125` | Last step Phase 2 |

---

## Phase 2 — Inside `build_cmapss_dataset()` (train/test Parquet)

```mermaid
flowchart TD
  START(["build_cmapss_dataset(FD00X)"])
  LOAD["Load raw\nload_cmapss_train / test / rul\nsrc/ingestion/cmapss_loader.py"]
  CFG2["Load config\nconfigs/cmapss_FD00X.yaml"]

  subgraph LABELS["Step A — Label engineering"]
    L1["Train RUL:\nmin(T - cycle, 125)"]
    L2["Test RUL:\nR_u + (t_max - cycle)"]
    L3["failure_30 = 1 if RUL ≤ 30\nfailure_72 = 1 if RUL ≤ 72"]
  end

  subgraph PRE["Step B — Preprocessing\nCmapssPreprocessor\nfit on TRAIN only"]
    P1["Drop sensors from config"]
    P2["Op clusters KMeans (FD002/004)\n→ op_cluster column"]
    P3["Per-unit baseline z-score\nfirst 5 cycles per engine"]
    P4["Cluster StandardScaler (FD002/004)"]
  end

  subgraph FE["Step C — Feature engineering\nCmapssFeatureEngineer"]
    F1["Rolling mean/std\nwindows 5, 10, 30"]
    F2["Lags 1, 3, 5"]
    F3["Delta cycle-to-cycle"]
    F4["Rolling slope per window"]
    F5["Spectral power RFFT\n5 sensors, window 10"]
    F6["degradation_index"]
  end

  FILTER["Step D — Train filter\nkeep rows where rul < 125"]
  WRITE["Write outputs"]

  subgraph PARQUET["Parquet & artifacts"]
    OUT1["cmapss_FD00X_train.parquet\nfiltered degrading rows"]
    OUT2["cmapss_FD00X_test.parquet\nall test cycles"]
    OUT3["preprocessor.joblib"]
    OUT4["feature_columns.json\n~188 cols FD001"]
  end

  START --> CFG2 --> LOAD --> LABELS
  LABELS --> PRE --> FE --> FILTER
  FE --> OUT2
  FILTER --> WRITE --> OUT1 & OUT3 & OUT4
```

### Columns in Phase 2 Parquet (conceptual)

| Column group | Examples | Role |
|--------------|----------|------|
| Keys | `unit_id`, `cycle` | Engine + time |
| Labels | `rul`, `failure_30`, `failure_72` | Train targets |
| Ops | `op_setting_1..3`, `op_cluster` | Regime context |
| Engineered | `sensor_2_roll5_mean`, `sensor_3_delta1`, … | Model inputs |
| Scalar | `degradation_index` | Health proxy feature |
| Raw sensors | Still in frame internally | Excluded from `feature_columns.json` list |

**Note:** Test parquet is **not** RUL-filtered (all censored trajectories kept for charts + last-cycle scoring).

---

## Phase 3 — Models → predictions Parquet

```mermaid
flowchart TD
  IN2["Read Phase 2 outputs"]
  TR_PQ["cmapss_FD00X_train.parquet"]
  TE_PQ["cmapss_FD00X_test.parquet"]
  FCOL["feature_columns.json"]

  SPLIT["Split train engines 80/20\nby unit_id"]

  subgraph RULM["RUL models — compare on validation"]
    RF["Random Forest regressor"]
    GBM["Gradient Boosting regressor"]
    LSTM["LSTM 30-cycle windows"]
    WIN["Pick winner\nlowest NASA score"]
  end

  subgraph AUX["Auxiliary models — all kept"]
    FC30["GBM failure_30"]
    FC72["GBM failure_72"]
    IF["Isolation Forest\nfit RUL ≥ 30 healthy rows"]
    COX["Cox PH survival\noptional baseline"]
  end

  RETRAIN["Retrain winner + aux on full train"]
  LAST["Last cycle per test engine"]
  TH["ThresholdEngine\nhealth, alert_level,\nrecommended_action"]
  EXPORT["build_fleet_predictions()"]

  subgraph OUTP["Outputs"]
    PRED["cmapss_FD00X_predictions.parquet\n1 row per engine"]
    PKL["models/rul_*, failure_*, anomaly_*, survival_*"]
    SUM2["cmapss_FD00X_phase3_summary.json"]
    REG2["cmapss_training_registry.json"]
  end

  IN2 --> TR_PQ & TE_PQ & FCOL
  TR_PQ --> SPLIT --> RULM --> WIN
  TR_PQ --> AUX
  WIN --> RETRAIN
  AUX --> RETRAIN
  TE_PQ --> LAST
  RETRAIN --> LAST --> TH --> EXPORT --> PRED
  RETRAIN --> PKL & SUM2 & REG2
```

### `predictions.parquet` — one row per engine (dashboard source)

| Column | Source model / logic |
|--------|-------------------|
| `asset_id`, `unit_id`, `cycle` | Last test row |
| `rul_true` | Label at last cycle |
| `rul_pred` | **RUL winner** (RF/GBM/LSTM) |
| `failure_prob_30`, `failure_prob_72` | Failure GBM classifiers |
| `rul_pred_cox`, `survival_prob_30/72` | Cox (if trained) |
| `anomaly_score`, `is_anomaly` | Isolation Forest |
| `health_score`, `risk_score` | ThresholdEngine formula |
| `alert_level`, `alert_message` | ThresholdEngine rules |
| `recommended_action`, `escalation_tier` | Rules + CMMS routing |
| `sensor_readings_json` | Snapshot at last cycle |
| `rul_model` | Winner name (`gbm`, `rf`, `lstm`) |

**File path:** `data/processed/cmapss_{FD001|FD002|FD003|FD004}_predictions.parquet`

---

## Complete file inventory (by folder)

```mermaid
flowchart LR
  subgraph raw["data/raw/cmapss/"]
    r1["train_FD00X.txt"]
    r2["test_FD00X.txt"]
    r3["RUL_FD00X.txt"]
  end

  subgraph proc["data/processed/"]
    p1["cmapss_FD00X_train.parquet"]
    p2["cmapss_FD00X_test.parquet"]
    p3["cmapss_FD00X_predictions.parquet"]
  end

  subgraph art["artifacts/"]
    a1["cmapss_FD00X_preprocessor.joblib"]
    a2["cmapss_FD00X_feature_columns.json"]
    a3["cmapss_FD00X_phase3_summary.json"]
    a4["cmapss_training_registry.json"]
  end

  subgraph cfg["configs/"]
    c1["cmapss_FD00X.yaml"]
    c2["cmapss_eda_snapshot.json"]
  end

  subgraph mod["models/"]
    m1["rul_{rf|gbm}_FD00X.pkl"]
    m2["rul_lstm_FD00X.pt"]
    m3["failure_30_FD00X.pkl"]
    m4["failure_72_FD00X.pkl"]
    m5["anomaly_FD00X.pkl"]
    m6["survival_FD00X.pkl"]
  end

  raw --> proc
  cfg --> proc
  proc --> mod
  mod --> p3
  art -.-> proc
  art -.-> mod
```

---

## Swimlane: who reads which Parquet

| Parquet | Rows | Primary consumer |
|---------|------|------------------|
| `*_train.parquet` | Many (degrading) | Phase 3 training only |
| `*_test.parquet` | Many (all cycles) | Phase 3 test metrics; Asset Detail **trajectory charts** |
| `*_predictions.parquet` | One per engine | Fleet Overview, Active Alerts, Asset Detail **summary**, API `/fleet` |

```mermaid
flowchart LR
  TR["train.parquet"] --> P3T["Phase 3 fit"]
  TE["test.parquet"] --> P3E["Phase 3 evaluate\n+ trajectory charts"]
  PR["predictions.parquet"] --> FLEET["Fleet / Alerts / API"]
  PR --> ASSET["Asset summary row"]
  TE --> CHART["RUL & sensor time series"]
```

---

## Code map (quick reference)

| Step | Script | Core module |
|------|--------|-------------|
| Download | `scripts/download_cmapss_data.py` | `src/ingestion/cmapss_download.py` |
| Phase 1 EDA | `scripts/run_cmapss_eda.py` | `src/ingestion/cmapss_eda.py` |
| Phase 2 build | `scripts/build_cmapss_dataset.py` | `src/ingestion/cmapss_pipeline.py` |
| Load/labels | — | `src/ingestion/cmapss_loader.py` |
| Preprocess | — | `src/ingestion/cmapss_preprocessor.py` |
| Features | — | `src/ingestion/feature_engineer.py` |
| Phase 3 | `scripts/train_cmapss_phase3.py` | `src/models/cmapss_phase3.py` |
| Re-export fleet | `scripts/export_fleet_predictions.py` | loads saved models + test parquet |
| Dashboard load | — | `dashboard/data_loader.py` |
| API load | — | `src/services/fleet_service.py` |

---

## Related docs

- [cmapss_eda_summary.md](cmapss_eda_summary.md) — Phase 1 findings  
- [cmapss_phase2_preprocessing.md](cmapss_phase2_preprocessing.md) — Phase 2 methodology  
- [cmapss_phase3_modeling.md](cmapss_phase3_modeling.md) — Phase 3 metrics & winner  
- [datasets/cmapss.md](datasets/cmapss.md) — NASA dataset overview  
