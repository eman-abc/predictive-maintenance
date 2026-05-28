#!/usr/bin/env python
"""Build processed CMAPSS train/test Parquet files (Phase 2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.cmapss_pipeline import build_cmapss_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CMAPSS Phase 2 datasets")
    parser.add_argument(
        "--dataset",
        default="FD001",
        help="CMAPSS subset (FD001, FD002, FD003, FD004)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all four FD subsets",
    )
    args = parser.parse_args()

    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.all else [args.dataset]
    for ds in datasets:
        result = build_cmapss_dataset(ds)
        meta = result["meta"]
        print(
            f"{ds}: train={meta['train_rows_filtered']:,} rows "
            f"(from {meta['train_rows_raw']:,}), test={meta['test_rows']:,}, "
            f"features={len(meta['feature_columns'])}"
        )
        print(f"  -> {result['train_path']}")
        print(f"  -> {result['test_path']}")


if __name__ == "__main__":
    main()
