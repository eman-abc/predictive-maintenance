#!/usr/bin/env python
"""Run CMAPSS Phase 3: model comparison, test evaluation, fleet predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cmapss_phase3 import run_phase3  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS Phase 3 training pipeline")
    parser.add_argument("--dataset", default=None, help="Single subset (FD001–FD004)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Multiple subsets, e.g. --datasets FD001 FD003",
    )
    parser.add_argument("--lstm-epochs", type=int, default=15)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    datasets = args.datasets or ([args.dataset] if args.dataset else ["FD001"])
    summaries = {}
    for ds in datasets:
        summaries[ds] = run_phase3(
            ds,
            val_fraction=args.val_fraction,
            lstm_epochs=args.lstm_epochs,
        )
    print(json.dumps(summaries if len(summaries) > 1 else summaries[datasets[0]], indent=2))


if __name__ == "__main__":
    main()
