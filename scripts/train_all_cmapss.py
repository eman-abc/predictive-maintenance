#!/usr/bin/env python
"""Build (optional) and train all CMAPSS FD001–FD004 subsets with MLflow logging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.cmapss_config import CMAPSS_DATASET_IDS  # noqa: E402
from src.ingestion.cmapss_pipeline import build_cmapss_dataset  # noqa: E402
from src.models.cmapss_phase3 import run_phase3_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all NASA CMAPSS subsets (Phase 2 build + Phase 3 MLflow)"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Assume processed Parquet already exists",
    )
    parser.add_argument("--lstm-epochs", type=int, default=15)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--gbm-max-rows", type=int, default=None)
    parser.add_argument("--anomaly-max-rows", type=int, default=None)
    args = parser.parse_args()

    for ds in CMAPSS_DATASET_IDS:
        if not args.skip_build:
            print(f"\n=== [{ds}] Phase 2: build dataset ===", flush=True)
            build_cmapss_dataset(ds)

    run_phase3_all(
        CMAPSS_DATASET_IDS,
        val_fraction=args.val_fraction,
        lstm_epochs=args.lstm_epochs,
        skip_lstm=args.skip_lstm,
        gbm_max_rows=args.gbm_max_rows,
        anomaly_max_rows=args.anomaly_max_rows,
    )

    print("\nVerification report:")
    print("  python scripts/report_cmapss_mlflow.py")
    print("\nMLflow UI:")
    print("  mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()
