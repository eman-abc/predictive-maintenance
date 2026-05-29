"""Load and parse NASA CMAPSS turbofan engine degradation datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

CMAPSS_COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


def _read_space_separated(path: Path) -> pd.DataFrame:
    """Read a CMAPSS space-delimited file into a DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    sensor_cols = [c for c in CMAPSS_COLUMNS if c.startswith("sensor_")]
    df[sensor_cols] = df[sensor_cols].astype(np.float64)
    return df


def load_cmapss_train(data_dir: str | Path, dataset: str = "FD001") -> pd.DataFrame:
    """Load CMAPSS training data (e.g. train_FD001.txt)."""
    path = Path(data_dir) / f"train_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"CMAPSS training file not found: {path}. "
            "Download from NASA Prognostics Data Repository."
        )
    return _read_space_separated(path)


def load_cmapss_test(data_dir: str | Path, dataset: str = "FD001") -> pd.DataFrame:
    """Load CMAPSS test data (e.g. test_FD001.txt)."""
    path = Path(data_dir) / f"test_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"CMAPSS test file not found: {path}")
    return _read_space_separated(path)


def load_cmapss_rul(data_dir: str | Path, dataset: str = "FD001") -> pd.Series:
    """Load ground-truth RUL labels for the test set."""
    path = Path(data_dir) / f"RUL_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"CMAPSS RUL file not found: {path}")
    rul = pd.read_csv(path, header=None, names=["rul"])
    return rul["rul"]


def compute_train_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Compute capped piecewise RUL for training trajectories (run-to-failure)."""
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df


def compute_test_rul(
    df: pd.DataFrame,
    rul_at_last_cycle: pd.Series,
    cap: int | None = 125,
) -> pd.DataFrame:
    """
    Reconstruct RUL at every test cycle from NASA's end-of-trajectory labels.

    For unit u with last observed cycle t_max and label R_u from RUL_FD00X.txt:
        RUL(t) = R_u + (t_max - t)

    Rows are matched to labels by sorted unit_id (official CMAPSS file order).
    """
    df = df.copy()
    units = sorted(df["unit_id"].unique())
    labels = rul_at_last_cycle.reset_index(drop=True)
    if len(units) != len(labels):
        raise ValueError(
            f"Test units ({len(units)}) and RUL labels ({len(labels)}) length mismatch"
        )
    rul_end = pd.Series(labels.values, index=units)
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = df["unit_id"].map(rul_end) + (max_cycle - df["cycle"])
    if cap is not None:
        df["rul"] = df["rul"].clip(upper=cap)
    return df


def add_failure_horizon_labels(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    rul_col: str = "rul",
) -> pd.DataFrame:
    """Add binary failure-within-horizon columns from RUL (e.g. failure_30)."""
    df = df.copy()
    for h in horizons or [30, 72]:
        df[f"failure_{h}"] = (df[rul_col] <= h).astype(int)
    return df
