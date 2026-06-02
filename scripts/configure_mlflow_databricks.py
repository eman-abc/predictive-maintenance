#!/usr/bin/env python
"""Set environment variables so MLflow logs to Databricks (Colab or local)."""

from __future__ import annotations

import argparse
import os
import sys


def configure(
    host: str,
    token: str,
    experiment: str = "/Shared/predictive_maintenance",
) -> None:
    host = host.rstrip("/")
    if not host.startswith("https://"):
        raise ValueError("DATABRICKS_HOST must start with https://")
    if not token or token.strip() == "":
        raise ValueError("DATABRICKS_TOKEN is empty")

    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token.strip()
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment


def smoke_test() -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
    with mlflow.start_run(run_name="colab_smoke_test") as run:
        mlflow.log_param("source", "colab")
        mlflow.log_metric("ok", 1.0)
    print("Smoke test run logged:", run.info.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure MLflow → Databricks")
    parser.add_argument("--host", required=True, help="Workspace URL, no trailing path")
    parser.add_argument("--token", required=True, help="PAT with mlflow scope")
    parser.add_argument(
        "--experiment",
        default="/Shared/predictive_maintenance",
        help="Databricks experiment path",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Log one test run")
    args = parser.parse_args()

    configure(args.host, args.token, args.experiment)
    print("MLFLOW_TRACKING_URI =", os.environ["MLFLOW_TRACKING_URI"])
    print("MLFLOW_EXPERIMENT_NAME =", os.environ["MLFLOW_EXPERIMENT_NAME"])
    print("DATABRICKS_HOST =", os.environ["DATABRICKS_HOST"])

    if args.smoke_test:
        smoke_test()


if __name__ == "__main__":
    main()
