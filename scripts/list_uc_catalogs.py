#!/usr/bin/env python
"""Print Unity Catalogs and schemas visible to your Databricks token."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.utils.databricks_uc import list_uc_catalogs, list_uc_schemas  # noqa: E402


def main() -> None:
    host = os.getenv("DATABRICKS_HOST", "")
    token = os.getenv("DATABRICKS_TOKEN", "")
    if not host or not token:
        print("Set DATABRICKS_HOST and DATABRICKS_TOKEN")
        sys.exit(1)

    catalogs = list_uc_catalogs(host, token)
    print("Catalogs:")
    for c in catalogs:
        print(f"  - {c}")
        try:
            for s in list_uc_schemas(host, token, c):
                print(f"      schema: {s}")
        except Exception as exc:
            print(f"      (schemas error: {exc})")


if __name__ == "__main__":
    main()
