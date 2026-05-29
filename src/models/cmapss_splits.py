"""Engine-level train/validation splits for CMAPSS (no row leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_unit_ids(
    unit_ids: list[int] | np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    """Hold out a fraction of engines for validation."""
    units = np.array(sorted(set(unit_ids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val].tolist())
    train_units = set(units[n_val:].tolist())
    return train_units, val_units


def filter_by_units(df: pd.DataFrame, unit_ids: set[int]) -> pd.DataFrame:
    return df[df["unit_id"].isin(unit_ids)].copy()
