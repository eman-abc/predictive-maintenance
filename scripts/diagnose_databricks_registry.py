#!/usr/bin/env python
"""Test Databricks MLflow + Unity Catalog registry before registering CMAPSS models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.utils.databricks_uc import resolve_uc_catalog_schema  # noqa: E402


def main() -> None:
    host = os.getenv("DATABRICKS_HOST", "")
    token = os.getenv("DATABRICKS_TOKEN", "")
    if not host or not token:
        print("FAIL: Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env or environment.")
        sys.exit(1)

    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    print("Tracking:", mlflow.get_tracking_uri())
    print("Registry:", mlflow.get_registry_uri())
    print()

    try:
        catalog, schema, meta = resolve_uc_catalog_schema(host, token)
    except Exception as exc:
        print("FAIL: Could not resolve Unity Catalog location:", exc)
        print("\nIn Databricks UI: Catalog Explorer (left sidebar)")
        print("  Copy the catalog and schema names you see (NOT 'main' unless it exists).")
        print("  Then set:")
        print('    os.environ["MLFLOW_UC_CATALOG"] = "your_catalog"')
        print('    os.environ["MLFLOW_UC_SCHEMA"] = "your_schema"')
        sys.exit(1)

    if meta.get("catalog_requested") and meta["catalog_requested"] != catalog:
        print(
            f"NOTE: MLFLOW_UC_CATALOG={meta['catalog_requested']!r} does not exist. "
            f"Using {catalog!r} instead."
        )
    if meta.get("schema_requested") and meta["schema_requested"] != schema:
        print(
            f"NOTE: MLFLOW_UC_SCHEMA={meta['schema_requested']!r} not in catalog. "
            f"Using {schema!r} instead."
        )

    print("Catalogs visible:", meta["catalogs_available"])
    print(f"Schemas in {catalog}:", meta["schemas_available"])
    print()
    print("USE THESE (copy to Colab / .env):")
    print(f'  MLFLOW_UC_CATALOG="{catalog}"')
    print(f'  MLFLOW_UC_SCHEMA="{schema}"')
    print()

    os.environ["MLFLOW_UC_CATALOG"] = catalog
    os.environ["MLFLOW_UC_SCHEMA"] = schema

    exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/predictive_maintenance")
    mlflow.set_experiment(exp)
    uc_name = f"{catalog}.{schema}.cmapss_diagnostic_smoke"
    artifact_name = "cmapss_diagnostic_smoke"

    print(f"Test register: {uc_name}")

    try:
        from sklearn.linear_model import Ridge
        import pandas as pd

        model = Ridge().fit([[1.0], [2.0]], [1.0, 2.0])
        with mlflow.start_run(run_name="registry_diagnostic"):
            info = mlflow.sklearn.log_model(
                model,
                name=artifact_name,
                registered_model_name=uc_name,
                input_example=pd.DataFrame({"x": [1.0]}),
            )
        print("OK: log_model + register succeeded.")
        print("  model_uri:", info.model_uri)
        print("  version:", getattr(info, "registered_model_version", "?"))
        print(f"\nIn UI: Catalog Explorer → {catalog} → {schema} → Models")
        sys.exit(0)
    except Exception as exc:
        print("FAIL:", exc)
        print("\nTypical fixes:")
        print("  1. Create schema in Catalog Explorer if list was empty")
        print("  2. PAT needs USE CATALOG + USE SCHEMA + CREATE MODEL on that schema")
        print("  3. git pull origin main")
        sys.exit(1)


if __name__ == "__main__":
    main()
