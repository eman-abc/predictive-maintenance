#!/usr/bin/env python
"""Smoke-test Databricks SQL warehouse + CMMS Delta table (uses .env)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main() -> int:
    from src.alerts.cmms_databricks import (
        databricks_credentials,
        databricks_table_links,
        is_databricks_logging_configured,
        table_fqn,
    )

    print("CMMS Databricks connection test")
    print("-" * 40)

    if not is_databricks_logging_configured():
        print("FAIL: CMMS_LOG_TO_DATABRICKS, DATABRICKS_HOST, DATABRICKS_TOKEN,")
        print("      DATABRICKS_SQL_HTTP_PATH (or WAREHOUSE_ID), CMMS_DELTA_TABLE")
        return 1

    host, _token, http_path = databricks_credentials()
    fqn = table_fqn()
    print(f"Host:      {host}")
    print(f"HTTP path: {http_path}")
    print(f"Table:     {fqn}")

    try:
        from src.alerts.cmms_databricks import _connect

        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                row = cur.fetchone()
        print(f"SELECT 1:  OK ({row})")
    except Exception as exc:
        msg = str(exc)
        print(f"FAIL: {msg}")
        if "scopes: sql" in msg.lower() or "scope" in msg.lower():
            print()
            print("Fix: User Settings → Developer → Access tokens → Generate new token")
            print("     Enable scope: SQL (or All APIs). Update DATABRICKS_TOKEN in .env")
        return 1

    links = databricks_table_links(fqn)
    if links.get("explore_url"):
        print(f"Explorer:  {links['explore_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
