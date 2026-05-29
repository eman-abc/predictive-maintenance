#!/usr/bin/env python
"""
End-to-end CMAPSS pipeline for FD001 + FD003 (build → train → export predictions).

Usage:
  python scripts/run_e2e.py
  python scripts/run_e2e.py --skip-build --skip-train
  python scripts/run_e2e.py --datasets FD001 FD003 --lstm-epochs 15
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.cmapss_pipeline import build_cmapss_dataset  # noqa: E402
from src.models.cmapss_phase3 import run_phase3  # noqa: E402

DEFAULT_DATASETS = ("FD001", "FD003")


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

    for ds in args.datasets:
        if not args.skip_build:
            print(f"\n=== [{ds}] Phase 2: build dataset ===")
            build_cmapss_dataset(ds)
        if not args.skip_train:
            print(f"\n=== [{ds}] Phase 3: train + export ===")
            run_phase3(ds, val_fraction=args.val_fraction, lstm_epochs=args.lstm_epochs)
        else:
            print(f"\n=== [{ds}] Export predictions only ===")
            _export_dataset(ds)

    print("\nDone. Start UI:")
    print("  mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
