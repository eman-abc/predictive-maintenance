#!/usr/bin/env python
"""Import user-uploaded CMAPSS files from Colab disk into data/raw/cmapss/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.cmapss_download import import_cmapss_from_dir, missing_cmapss_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy CMAPSS txt files from an upload folder into the project"
    )
    parser.add_argument(
        "--source",
        default="/content/cmapss_upload",
        help="Folder where you uploaded train_*.txt, test_*.txt, RUL_*.txt",
    )
    parser.add_argument(
        "--dest",
        default=str(ROOT / "data" / "raw" / "cmapss"),
        help="Project raw data directory",
    )
    args = parser.parse_args()

    import_cmapss_from_dir(args.source, args.dest)
    missing = missing_cmapss_files(args.dest)
    if missing:
        raise SystemExit(f"Still missing: {missing}")
    print(f"Ready: {Path(args.dest).resolve()}")


if __name__ == "__main__":
    main()
