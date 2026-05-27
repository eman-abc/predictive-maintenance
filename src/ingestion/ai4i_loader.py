"""Load AI4I 2020 Predictive Maintenance dataset."""

from pathlib import Path

import pandas as pd


def load_ai4i(data_dir: str | Path, filename: str = "ai4i2020.csv") -> pd.DataFrame:
    """Load the AI4I 2020 CSV and derive a unified failure label."""
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"AI4I dataset not found: {path}. "
            "Download from UCI ML Repository or Kaggle."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    failure_cols = [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    existing = [c for c in failure_cols if c in df.columns]
    if "Machine failure" in df.columns:
        df["failure"] = df["Machine failure"].astype(int)
    elif existing:
        df["failure"] = df[existing].max(axis=1).astype(int)
    else:
        df["failure"] = 0

    return df
