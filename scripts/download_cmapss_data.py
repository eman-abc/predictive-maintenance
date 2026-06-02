#!/usr/bin/env python
"""Download CMAPSS FD001–FD004 raw files into data/raw/cmapss/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.cmapss_download import download_cmapss_raw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NASA CMAPSS raw data")
    parser.add_argument(
        "--dest",
        default=str(ROOT / "data" / "raw" / "cmapss"),
        help="Output directory",
    )
    parser.add_argument("--force", action="store_true", help="Re-download all files")
    args = parser.parse_args()
    download_cmapss_raw(args.dest, force=args.force)
    print(f"\nDone: {Path(args.dest).resolve()}")


if __name__ == "__main__":
    main()
