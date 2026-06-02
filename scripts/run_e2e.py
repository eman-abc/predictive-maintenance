#!/usr/bin/env python
"""
End-to-end CMAPSS pipeline for FD001–FD004 (build → train → export predictions).

Usage:
  python scripts/run_e2e.py
  python scripts/run_e2e.py --skip-build --skip-train
  python scripts/run_e2e.py --datasets FD001 FD002 --lstm-epochs 15
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.cmapss_config import CMAPSS_DATASET_IDS  # noqa: E402
from src.ingestion.cmapss_pipeline import build_cmapss_dataset  # noqa: E402
from src.models.cmapss_phase3 import run_phase3_all  # noqa: E402

DEFAULT_DATASETS = CMAPSS_DATASET_IDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full E2E CMAPSS pipeline")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="FD subsets to process",
    )
    parser.add_argument("--lstm-epochs", type=int, default=15)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--gbm-max-rows", type=int, default=None)
    parser.add_argument("--anomaly-max-rows", type=int, default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    def _export_dataset(ds: str) -> None:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_fleet_predictions.py"), "--dataset", ds],
            cwd=str(ROOT),
            check=True,
        )

    if args.export_only:
        for ds in args.datasets:
            _export_dataset(ds)
        return

    if not args.skip_build:
        for ds in args.datasets:
            print(f"\n=== [{ds}] Phase 2: build dataset ===")
            build_cmapss_dataset(ds)

    if not args.skip_train:
        print(f"\n=== Phase 3: train {', '.join(args.datasets)} ===")
        run_phase3_all(
            args.datasets,
            val_fraction=args.val_fraction,
            lstm_epochs=args.lstm_epochs,
            skip_lstm=args.skip_lstm,
            gbm_max_rows=args.gbm_max_rows,
            anomaly_max_rows=args.anomaly_max_rows,
        )
    else:
        for ds in args.datasets:
            print(f"\n=== [{ds}] Export predictions only ===")
            _export_dataset(ds)

    print("\nDone. Verify training:")
    print("  python scripts/report_cmapss_mlflow.py")
    print("  mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
