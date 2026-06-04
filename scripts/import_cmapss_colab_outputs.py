#!/usr/bin/env python
"""Import Colab zip contents (models/, artifacts/, data/processed/) into the project tree."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "cmapss_colab_outputs"

SUBDIRS = ("models", "artifacts", "data/processed")


def _copy_tree(source: Path, dest: Path, *, dry_run: bool, force: bool) -> tuple[int, int]:
    copied = skipped = 0
    if not source.is_dir():
        return copied, skipped

    dest.mkdir(parents=True, exist_ok=True)
    for src in sorted(source.iterdir()):
        if src.name.startswith(".") or src.name == ".gitkeep":
            continue
        dst = dest / src.name
        if dst.exists() and not force:
            skipped += 1
            continue
        if dry_run:
            print(f"  would copy {src.name} -> {dest}/")
            copied += 1
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied += 1
        print(f"  copied {src.name}")
    return copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Import cmapss_colab_outputs into project paths")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Unzipped Colab folder (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument("--dest", type=Path, default=ROOT, help="Project root")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        print(f"FAIL: source not found: {source}", file=sys.stderr)
        sys.exit(1)

    total_copied = total_skipped = 0
    for sub in SUBDIRS:
        src = source / sub
        dst = args.dest / sub
        print(f"{sub}/")
        c, s = _copy_tree(src, dst, dry_run=args.dry_run, force=args.force)
        total_copied += c
        total_skipped += s

    print(f"\nDone: {total_copied} copied, {total_skipped} skipped (use --force to overwrite)")
    if total_copied == 0 and total_skipped > 0:
        print("Tip: run with --force to refresh predictions, summaries, and survival models.")


if __name__ == "__main__":
    main()
