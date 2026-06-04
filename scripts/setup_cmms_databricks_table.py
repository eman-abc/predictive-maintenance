#!/usr/bin/env python
"""Create the CMMS work orders Delta table in Unity Catalog (one-time setup)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cmms_work_orders Delta table")
    parser.add_argument(
        "--print-sql-only",
        action="store_true",
        help="Print CREATE TABLE SQL for manual run in Databricks SQL editor",
    )
    args = parser.parse_args()

    from src.alerts.cmms_databricks import (
        auto_table_fqn,
        create_table_sql,
        ensure_table,
        explore_table_url,
        schema_fqn_from_table,
        table_fqn,
    )

    manual_fqn = table_fqn()
    auto_fqn = auto_table_fqn()

    if args.print_sql_only:
        print(create_table_sql(manual_fqn))
        print()
        print(create_table_sql(auto_fqn))
        return

    schema = schema_fqn_from_table(manual_fqn)
    print(f"Creating schema (if missing): {schema}")
    try:
        for label, fqn in [("manual", manual_fqn), ("auto-dispatch", auto_fqn)]:
            print(f"Creating table ({label}): {fqn}")
            ensure_table(fqn=fqn)
    except Exception as exc:
        print(f"\nFAIL: {exc}")
        print("\nIf catalog/schema is wrong, run:")
        print("  python scripts/diagnose_databricks_registry.py")
        print("Then set CMMS_DELTA_TABLE and CMMS_AUTO_DELTA_TABLE in .env")
        raise SystemExit(1) from exc
    print("Done.")
    for fqn in (manual_fqn, auto_fqn):
        url = explore_table_url(fqn)
        if url:
            print(f"Open {fqn}: {url}")
    print("\nEnable logging in .env:")
    print("  CMMS_LOG_TO_DATABRICKS=true")
    print("  CMMS_AUTO_DISPATCH=true")


if __name__ == "__main__":
    main()
