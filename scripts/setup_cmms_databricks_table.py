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
        create_schema_sql,
        create_table_sql,
        ensure_table,
        explore_table_url,
        schema_fqn_from_table,
        table_fqn,
    )

    fqn = table_fqn()
    sql = create_table_sql(fqn)

    if args.print_sql_only:
        print(sql)
        return

    schema = schema_fqn_from_table(fqn)
    print(f"Creating schema (if missing): {schema}")
    print(f"Creating table (if missing): {fqn}")
    try:
        ensure_table(fqn=fqn)
    except Exception as exc:
        print(f"\nFAIL: {exc}")
        print("\nIf catalog/schema is wrong, run:")
        print("  python scripts/diagnose_databricks_registry.py")
        print("Then set CMMS_DELTA_TABLE=<catalog>.<schema>.cmms_work_orders in .env")
        raise SystemExit(1) from exc
    print("Done.")
    url = explore_table_url(fqn)
    if url:
        print(f"Open in UI: {url}")
    print("\nEnable logging in .env:")
    print("  CMMS_LOG_TO_DATABRICKS=true")


if __name__ == "__main__":
    main()
