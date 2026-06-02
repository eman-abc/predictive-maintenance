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


def main() -> None:
    host = os.getenv("DATABRICKS_HOST", "")
    token = os.getenv("DATABRICKS_TOKEN", "")
    if not host or not token:
        print("FAIL: Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env or environment.")
        sys.exit(1)

    import mlflow
    from mlflow.tracking import MlflowClient

    catalog = os.getenv("MLFLOW_UC_CATALOG", "main")
    schema = os.getenv("MLFLOW_UC_SCHEMA", "default")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    print("Tracking:", mlflow.get_tracking_uri())
    print("Registry:", mlflow.get_registry_uri())
    print(f"UC target: {catalog}.{schema}.<model>")
    print()

    client = MlflowClient(registry_uri="databricks-uc")

    try:
        catalogs = [c.name for c in client.search_catalogs()]
        print("Catalogs you can see:", catalogs[:10])
        if catalog not in catalogs:
            print(f"WARN: MLFLOW_UC_CATALOG={catalog!r} not in list — pick one from above.")
    except Exception as exc:
        print("Could not list catalogs:", exc)

    try:
        schemas = [s.name for s in client.search_schemas(catalog)]
        print(f"Schemas in {catalog}:", schemas[:15])
        if schema not in schemas:
            print(f"WARN: MLFLOW_UC_SCHEMA={schema!r} not in list — pick one from above.")
    except Exception as exc:
        print(f"Could not list schemas in {catalog}:", exc)

    exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/predictive_maintenance")
    mlflow.set_experiment(exp)
    uc_name = f"{catalog}.{schema}.cmapss_diagnostic_smoke"
    artifact_name = "cmapss_diagnostic_smoke"

    print(f"\nTest register: {uc_name}")
    print(f"  log_model name (short): {artifact_name}")

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
        print(f"\nIn UI: Catalog → {catalog} → {schema} → Models → {artifact_name}")
        sys.exit(0)
    except Exception as exc:
        print("FAIL:", exc)
        print("\nTypical fixes:")
        print("  1. Set MLFLOW_UC_CATALOG / MLFLOW_UC_SCHEMA from Catalog Explorer")
        print("  2. PAT needs CREATE MODEL on that schema (or use a schema you own)")
        print("  3. git pull main (artifact name vs UC name fix)")
        sys.exit(1)


if __name__ == "__main__":
    main()
