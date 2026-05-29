# Predictive Maintenance Dataset (AI4I 2020)

## Overview
A synthetic dataset modeled after an existing milling machine, consisting of 10,000 data points with 14 features. The dataset is designed for classification and Explainable Artificial Intelligence (XAI) tasks in predictive maintenance.

**Objective:** Predict machine failure and/or identify the specific failure mode that caused the machine to break down.

## Data Dictionary
The dataset is provided as a single tabular file (`ai4i2020.csv`).

### Identifier & Product Features
| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `UID` | Integer | Unique identifier ranging from 1 to 10000. |
| `Product ID` | String | Consists of a letter (L, M, H) denoting quality variant and a variant-specific serial number. |
| `Type` | Categorical | Product quality variant: **L** (Low, 50%), **M** (Medium, 30%), **H** (High, 20%). |

### Sensor / Process Measurements
| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `Air temperature [K]` | Float | Generated via random walk, normalized to standard deviation of 2 K around 300 K. |
| `Process temperature [K]` | Float | Generated via random walk, normalized to standard deviation of 1 K, added to air temperature + 10 K. |
| `Rotational speed [rpm]` | Float | Calculated from a power of 2860 W, overlaid with normally distributed noise. |
| `Torque [Nm]` | Float | Normally distributed around 40 Nm (SD = 10 Nm, no negative values). |
| `Tool wear [min]` | Float | Quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. |

### Target Variables (Labels)
The primary target is `Machine failure`. A failure occurs (label = 1) if *at least one* of the five independent failure modes below is true. 

| Feature Name | Type | Description | Occurrence in Data |
| :--- | :--- | :--- | :--- |
| `Machine failure` | Binary (0/1) | General failure indicator. 1 if any specific failure mode below is triggered. | - |
| `TWF` | Binary (0/1) | **Tool Wear Failure:** Tool fails/is replaced at random wear time between 200 - 240 mins. | 120 times (51 true failures, 69 replacements) |
| `HDF` | Binary (0/1) | **Heat Dissipation Failure:** Difference between air and process temp < 8.6 K AND rotational speed < 1380 rpm. | 115 times |
| `PWF` | Binary (0/1) | **Power Failure:** Required power (torque * rad/s) is < 3500 W or > 9000 W. | 95 times |
| `OSF` | Binary (0/1) | **Overstrain Failure:** Tool wear * torque exceeds threshold (L: 11k, M: 12k, H: 13k minNm). | 98 times |
| `RNF` | Binary (0/1) | **Random Failure:** 0.1% chance to fail regardless of process parameters. | 5 times |