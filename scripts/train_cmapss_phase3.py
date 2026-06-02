#!/usr/bin/env python
"""Run CMAPSS Phase 3: model comparison, test evaluation, fleet predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.cmapss_config import CMAPSS_DATASET_IDS  # noqa: E402
from src.models.cmapss_phase3 import run_phase3, run_phase3_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS Phase 3 training pipeline")
    parser.add_argument("--dataset", default=None, help="Single subset (FD001–FD004)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Multiple subsets, e.g. --datasets FD001 FD003",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Train all subsets: {', '.join(CMAPSS_DATASET_IDS)}",
    )
    parser.add_argument("--lstm-epochs", type=int, default=15)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM (RF vs GBM only); much faster on CPU/Colab",
    )
    parser.add_argument(
        "--skip-cox",
        action="store_true",
        help="Skip Cox PH survival model (lifelines)",
    )
    parser.add_argument(
        "--gbm-max-rows",
        type=int,
        default=None,
        help=f"Max rows for GBM fit (default {250_000})",
    )
    parser.add_argument(
        "--anomaly-max-rows",
        type=int,
        default=None,
        help="Max rows for anomaly detector fit (default 100000)",
    )
    args = parser.parse_args()

    train_kw = dict(
        val_fraction=args.val_fraction,
        lstm_epochs=args.lstm_epochs,
        skip_lstm=args.skip_lstm,
        skip_cox=args.skip_cox,
        gbm_max_rows=args.gbm_max_rows,
        anomaly_max_rows=args.anomaly_max_rows,
    )

    if args.all:
        datasets = list(CMAPSS_DATASET_IDS)
    elif args.datasets:
        datasets = args.datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(CMAPSS_DATASET_IDS)

    if len(datasets) == 1:
        summaries = {datasets[0]: run_phase3(datasets[0], **train_kw)}
    else:
        summaries = run_phase3_all(datasets, **train_kw)
    print(json.dumps(summaries if len(summaries) > 1 else summaries[datasets[0]], indent=2))
    print("\nVerify MLflow + artifacts: python scripts/report_cmapss_mlflow.py", flush=True)


if __name__ == "__main__":
    main()
