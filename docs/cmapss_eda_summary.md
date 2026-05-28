# CMAPSS Phase 1 — Exploratory Data Analysis

This document records **methodology**, **findings**, and **downstream decisions** from Phase 1 (EDA & sensor QC) for NASA C-MAPSS FD001–FD004. Configs generated from this analysis live in `configs/cmapss_FD00X.yaml`.

## 1. Objectives

Phase 1 answers four questions before feature engineering:

1. **Fleet structure** — How many engines, cycles per engine, train vs test behavior?
2. **Sensor quality** — Which sensors are constant or uninformative?
3. **Operating conditions** — Are op settings static (FD001/003) or multi-condition (FD002/004)?
4. **Test censorship** — Test run lengths and RUL-at-last-cycle from official label files.

These decisions satisfy UC5 Component A prerequisites: justified data understanding before health indicators and target definition.

## 2. Methodology

### 2.1 Data source and schema

- **Source:** NASA C-MAPSS turbofan simulation ([readme.txt](../data/raw/cmapss/readme.txt), Saxena et al., PHM08).
- **Files per subset:** `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`.
- **Schema:** 26 space-separated columns per row (see `src/ingestion/cmapss_loader.py`).
- **Loader:** `pandas.read_csv(sep=r'\s+')` with explicit column names (no header row).

### 2.2 Trajectory statistics

For each subset and split (train/test):

- Count distinct `unit_id` (engines).
- Per unit, record `max(cycle)` → run length distribution (min, max, median).
- **Train:** trajectories run until simulated failure (RUL → 0 at last cycle).
- **Test:** trajectories stop before failure; `RUL_FD00X.txt` gives true RUL at the last observed cycle.

### 2.3 Train vs test trajectory behavior

**Important:** Train and test engines are **different units** (same fleet type, not the same engines). We do **not** subtract train run length from test run length by `unit_id`.

Instead we report:

1. **Train run length** — cycles until simulated failure (RUL→0 at last row).
2. **Test run length** — cycles observed before censorship.
3. **RUL at last test cycle** — from `RUL_FD00X.txt` (competition ground truth).
4. **Implied failure cycle** — `last_test_cycle + RUL` per test engine (sanity check on label file).

### 2.4 Sensor variance QC

On **train** and **test** separately, compute standard deviation per `sensor_1`…`sensor_21`:

| Classification | Rule |
|----------------|------|
| **Constant** | `std == 0` |
| **Near-constant** | `0 < std < 1e-6` |
| **Informative** | `std ≥ 1e-6` |

**Drop recommendation** = union of constant and near-constant sensors across **both** splits.
For FD001, we also union the literature prior (Saxena et al.; common benchmark practice).

### 2.5 Operating settings

NASA labels each subset with a **scenario** count (1 or 6 operating conditions). Per-column `nunique` is high even for FD001 because settings differ across engines.

We therefore report:

- **Per-column nunique** — diagnostic only.
- **Unique (op1, op2, op3) triplets** (rounded to 2 dp) — how many distinct setting vectors appear.
- **Per-unit stability** — whether each engine holds fixed settings across its life.
- **`cluster_for_normalization`** — `true` when NASA scenario has 6 conditions (FD002, FD004).

### 2.6 Reproducibility

- Script: `scripts/run_cmapss_eda.py`
- Library: `src/ingestion/cmapss_eda.py`
- Re-run: `python scripts/run_cmapss_eda.py`

## 3. Decisions (locked for Phase 2+)

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | **Primary target:** piecewise capped RUL (`cap=125`) | NASA benchmark convention; reduces healthy-plateau noise |
| D2 | **Secondary targets:** `failure_30`, `failure_72` (`rul ≤ horizon`) | Maps UC5 failure-window requirement to cycle-based labels |
| D3 | **Drop constant sensors** per EDA table below | Zero variance adds no information; reduces overfitting |
| D4 | **FD001 first**, then FD003 (2 fault modes), then FD002/004 | Complexity ladder; UC5 requires ≥2 failure modes (FD003/004) |
| D5 | **No row-level random split** | Official split is by engine; leakage if rows are shuffled across units |
| D6 | **Op-setting clustering** for FD002/FD004 (6 NASA scenarios) | Condition-aware normalization when multiple flight regimes exist |
| D7 | **Rolling windows** `[5, 10, 30]`, **lags** `[1, 3, 5]` | Short/medium/long degradation horizons for ~100–300 cycle runs |
| D8 | **Train rows for RUL modeling:** focus on `rul ≤ 125` (Phase 4) | Standard practice; defers to config `train_row_filter_max_rul` |

## 4. Results by dataset

### FD001

| Property | Value |
|----------|-------|
| Operating conditions | 1 |
| Fault modes | 1 |
| Train engines × rows | 100 × 20,631 |
| Test engines × rows | 100 × 13,096 |
| Train run length (min / median / max) | 128 / 199 / 362 |
| Test run length (min / median / max) | 31 / 134 / 303 |
| RUL at last test cycle (min / median / max) | 7 / 86 / 145 |
| Implied failure cycle (median) | 199 |
| NASA op scenarios | 1 |
| Unique op triplets (train) | 3 |
| Op settings stable within each unit? | **No** |
| Cluster for normalization (Phase 3)? | **No** |

**Sensors to drop:** `sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`

**Sensors to keep:** `sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`

*Train-only constant:* `sensor_1`, `sensor_10`, `sensor_18`, `sensor_19`  
*Test-only constant:* `sensor_1`, `sensor_18`, `sensor_19`  

### FD002

| Property | Value |
|----------|-------|
| Operating conditions | 6 |
| Fault modes | 1 |
| Train engines × rows | 260 × 53,759 |
| Test engines × rows | 259 × 33,991 |
| Train run length (min / median / max) | 128 / 199 / 378 |
| Test run length (min / median / max) | 21 / 132 / 367 |
| RUL at last test cycle (min / median / max) | 6 / 80 / 194 |
| Implied failure cycle (median) | 204 |
| NASA op scenarios | 6 |
| Unique op triplets (train) | 11 |
| Op settings stable within each unit? | **No** |
| Cluster for normalization (Phase 3)? | **Yes** |

**Sensors to drop:** _none_

**Sensors to keep:** `sensor_1`, `sensor_2`, `sensor_3`, `sensor_4`, `sensor_5`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_10`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_16`, `sensor_17`, `sensor_18`, `sensor_19`, `sensor_20`, `sensor_21`

*Train-only constant:* _none_  
*Test-only constant:* _none_  

### FD003

| Property | Value |
|----------|-------|
| Operating conditions | 1 |
| Fault modes | 2 |
| Train engines × rows | 100 × 24,720 |
| Test engines × rows | 100 × 16,596 |
| Train run length (min / median / max) | 145 / 220 / 525 |
| Test run length (min / median / max) | 38 / 148 / 475 |
| RUL at last test cycle (min / median / max) | 6 / 78 / 145 |
| Implied failure cycle (median) | 222 |
| NASA op scenarios | 1 |
| Unique op triplets (train) | 3 |
| Op settings stable within each unit? | **No** |
| Cluster for normalization (Phase 3)? | **No** |

**Sensors to drop:** `sensor_1`, `sensor_5`, `sensor_16`, `sensor_18`, `sensor_19`

**Sensors to keep:** `sensor_2`, `sensor_3`, `sensor_4`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_10`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`

*Train-only constant:* `sensor_1`, `sensor_18`, `sensor_19`  
*Test-only constant:* `sensor_18`, `sensor_19`  

### FD004

| Property | Value |
|----------|-------|
| Operating conditions | 6 |
| Fault modes | 2 |
| Train engines × rows | 249 × 61,249 |
| Test engines × rows | 248 × 41,214 |
| Train run length (min / median / max) | 128 / 234 / 543 |
| Test run length (min / median / max) | 19 / 154 / 486 |
| RUL at last test cycle (min / median / max) | 6 / 88 / 195 |
| Implied failure cycle (median) | 235 |
| NASA op scenarios | 6 |
| Unique op triplets (train) | 11 |
| Op settings stable within each unit? | **No** |
| Cluster for normalization (Phase 3)? | **Yes** |

**Sensors to drop:** _none_

**Sensors to keep:** `sensor_1`, `sensor_2`, `sensor_3`, `sensor_4`, `sensor_5`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_10`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_16`, `sensor_17`, `sensor_18`, `sensor_19`, `sensor_20`, `sensor_21`

*Train-only constant:* _none_  
*Test-only constant:* _none_  

## 5. Cross-dataset comparison

| Dataset | Fault modes | Op conditions | Keep sensors | Typical use |
|---------|-------------|---------------|--------------|-------------|
| FD001 | 1 | 1 | 14 | Baseline |
| FD002 | 1 | 6 | 21 | Multi-condition |
| FD003 | 2 | 1 | 16 | Multi-mode |
| FD004 | 2 | 6 | 21 | Multi-mode |

## 6. UC5 alignment

| UC5 requirement | Phase 1 outcome |
|-----------------|-----------------|
| Multi-unit time series | Confirmed 100–260 engines per subset |
| ≥ 2 failure modes | **FD003, FD004** (2 modes); FD001/002 for ablation |
| Justified preprocessing | Sensor drop lists + op-condition flags per subset |
| RUL target | `rul_cap: 125` in configs; test RUL file validated |

## 7. Next steps (Phase 2)

1. Implement `compute_test_rul()` — align `RUL_FD00X.txt` to every test cycle.
2. Build leakage-safe normalization (per unit; cluster for FD002/004).
3. Add delta, slope, and spectral features per config.
4. Persist `data/processed/cmapss_{FD00X}_{train,test}_features.parquet`.

## 8. References

- Saxena, A., et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* PHM08.
- NASA C-MAPSS readme: `data/raw/cmapss/readme.txt`
- Dataset overview: `docs/datasets/cmapss.md`
