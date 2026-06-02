#!/usr/bin/env python
"""
Register on-disk CMAPSS models into MLflow Model Registry (Databricks or local).

Use after Colab training if registration was skipped, or to re-register from a zip.

  python scripts/register_cmapss_models.py --datasets FD001 FD002 FD003 FD004
  python scripts/register_cmapss_models.py --dataset FD001 --run-label uc5_reregister_v1

Requires MLFLOW_TRACKING_URI (e.g. databricks) and credentials in the environment.
Set MLFLOW_REGISTER_MODELS=0 to dry-run (script exits without registering).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.ingestion.cmapss_config import CMAPSS_DATASET_IDS  # noqa: E402
from src.models.mlflow_registry import (  # noqa: E402
    register_models_from_disk,
    registry_enabled,
    setup_model_registry_uri,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register CMAPSS models in MLflow Model Registry")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--run-label",
        default=None,
        help="training_batch tag on the registration run",
    )
    parser.add_argument(
        "--mlflow-databricks",
        action="store_true",
        help="Require DATABRICKS_HOST + DATABRICKS_TOKEN",
    )
    args = parser.parse_args()

    if args.mlflow_databricks:
        if not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"):
            print("Set DATABRICKS_HOST and DATABRICKS_TOKEN", file=sys.stderr)
            sys.exit(1)
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"

    if args.run_label:
        os.environ["MLFLOW_TRAINING_BATCH_ID"] = args.run_label

    if not registry_enabled():
        print("MLFLOW_REGISTER_MODELS=0 — nothing to do.")
        sys.exit(0)

    if args.all:
        datasets = list(CMAPSS_DATASET_IDS)
    elif args.datasets:
        datasets = args.datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(CMAPSS_DATASET_IDS)

    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance"))
    reg_uri = setup_model_registry_uri()
    print(f"MLflow registry URI: {reg_uri}", flush=True)
    if reg_uri == "databricks-uc":
        cat = os.getenv("MLFLOW_UC_CATALOG", "main")
        sch = os.getenv("MLFLOW_UC_SCHEMA", "default")
        print(f"UC model names: {cat}.{sch}.cmapss_<role>_FD00X", flush=True)

    all_registered: dict[str, dict] = {}
    for ds in datasets:
        print(f"\n=== Register {ds} ===", flush=True)
        all_registered[ds] = register_models_from_disk(ds, training_batch=args.run_label)

    print("\nDone. In Databricks: Machine Learning → Models", flush=True)
    for ds, reg in all_registered.items():
        if reg:
            print(f"  {ds}: {', '.join(v['name'] for v in reg.values())}")
        else:
            print(f"  {ds}: (no models registered — check models/ and summary JSON)")


if __name__ == "__main__":
    main()
