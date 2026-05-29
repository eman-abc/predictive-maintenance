# NASA C-MAPSS-1 Turbofan Engine Degradation Dataset

## Overview
This dataset contains comprehensive prognostic data for turbofan engine degradation simulation, generated using NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS). It provides run-to-failure time series data capturing the degradation patterns of aircraft engines under various operational conditions.

**Objective:** Predict the Remaining Useful Life (RUL) of the engine (the number of operational cycles before failure in the test set).

## File Structure
The dataset is split into four distinct sub-datasets (FD001 to FD004) representing different operational scenarios:

* **`train_FD00X.txt`**: Training data. Fault grows in magnitude until system failure.
* **`test_FD00X.txt`**: Testing data. Time series ends some time prior to system failure.
* **`RUL_FD00X.txt`**: Ground truth files containing the true remaining useful life (RUL) values for each test engine.

### Scenario Details
| Dataset | Train Trajectories | Test Trajectories | Operating Conditions | Fault Modes | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FD001** | 100 | 100 | 1 (Sea Level) | 1 (HPC Degradation) | Baseline scenario. Good for initial model development. |
| **FD002** | 260 | 259 | 6 | 1 (HPC Degradation) | Introduces varying operational conditions. |
| **FD003** | 100 | 100 | 1 (Sea Level) | 2 (HPC & Fan Degradation) | Introduces multiple fault modes. |
| **FD004** | 248 | 249 | 6 | 2 (HPC & Fan Degradation) | Most complex. Multiple conditions and fault modes. |

## Data Dictionary
The training and test data are provided as text files with 26 space-separated columns. Each row is a snapshot of data taken during a single operational cycle.

| Column Index | Feature Name | Description |
| :--- | :--- | :--- |
| 1 | `unit_number` | Engine unit identifier. |
| 2 | `time_cycles` | Operational cycle number (time). |
| 3 | `op_setting_1` | Operational setting 1 (e.g., flight altitude). |
| 4 | `op_setting_2` | Operational setting 2 (e.g., Mach number). |
| 5 | `op_setting_3` | Operational setting 3 (e.g., throttle resolver angle). |
| 6-26 | `sensor_1` to `sensor_21` | 21 distinct sensor measurements (temperatures, pressures, speeds, ratios) characterizing engine health. Contaminated with sensor noise. |

## Phase 3 modeling

- **Report:** [cmapss_phase3_modeling.md](../cmapss_phase3_modeling.md)
- **Train:** `python scripts/train_cmapss_phase3.py --dataset FD001`

## Phase 2 preprocessing (completed)

- **Report:** [cmapss_phase2_preprocessing.md](../cmapss_phase2_preprocessing.md)
- **Build:** `python scripts/build_cmapss_dataset.py --dataset FD001`

## Phase 1 EDA (completed)

Full methodology, sensor keep/drop decisions, and pipeline configs:

- **Report:** [cmapss_eda_summary.md](../cmapss_eda_summary.md)
- **Configs:** `configs/cmapss_FD001.yaml` … `configs/cmapss_FD004.yaml`
- **Re-run:** `python scripts/run_cmapss_eda.py`