#!/usr/bin/env python
"""
Colab / cloud runner: build CMAPSS features, train all FD subsets, verify MLflow.

Typical Colab (Runtime → GPU optional for LSTM):

  %cd /content/predictive-maintenance
  !python scripts/cmapss_colab_train.py --fast

Full quality (slower):

  !python scripts/cmapss_colab_train.py

Environment overrides:
  CMAPSS_COLAB_SKIP_BUILD=1   skip Phase 2 if Parquet exists
  CMAPSS_COLAB_SKIP_LSTM=1    same as --skip-lstm
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.cmapss_config import CMAPSS_DATASET_IDS  # noqa: E402
from src.ingestion.cmapss_pipeline import build_cmapss_dataset  # noqa: E402
from src.models.cmapss_phase3 import run_phase3_all  # noqa: E402


def _cuda_info() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        return "CUDA: not available (LSTM runs on CPU)"
    except ImportError:
        return "PyTorch not installed"


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS Colab/cloud training runner")
    parser.add_argument("--fast", action="store_true", help="skip-lstm + smaller GBM subsample")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--lstm-epochs", type=int, default=15)
    parser.add_argument("--gbm-max-rows", type=int, default=None)
    parser.add_argument("--anomaly-max-rows", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(CMAPSS_DATASET_IDS))
    parser.add_argument(
        "--upload-dir",
        default=None,
        help="Colab folder with uploaded CMAPSS txt files (e.g. /content/cmapss_upload)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download from mirror if raw files missing (default: require upload)",
    )
    parser.add_argument(
        "--mlflow-databricks",
        action="store_true",
        help="Require DATABRICKS_HOST + DATABRICKS_TOKEN in environment",
    )
    args = parser.parse_args()

    if os.getenv("CMAPSS_COLAB_SKIP_BUILD", "").lower() in ("1", "true", "yes"):
        args.skip_build = True
    if os.getenv("CMAPSS_COLAB_SKIP_LSTM", "").lower() in ("1", "true", "yes"):
        args.skip_lstm = True

    skip_lstm = args.skip_lstm or args.fast
    gbm_max = args.gbm_max_rows or (100_000 if args.fast else None)
    anomaly_max = args.anomaly_max_rows or (50_000 if args.fast else None)
    lstm_epochs = 5 if args.fast else args.lstm_epochs

    os.chdir(ROOT)

    tracking = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    if args.mlflow_databricks or tracking == "databricks":
        host = os.getenv("DATABRICKS_HOST", "")
        token = os.getenv("DATABRICKS_TOKEN", "")
        if not host or not token:
            print(
                "ERROR: Set DATABRICKS_HOST and DATABRICKS_TOKEN before training.\n"
                "  python scripts/configure_mlflow_databricks.py --host https://... "
                "--token <PAT> --smoke-test",
                file=sys.stderr,
            )
            sys.exit(1)
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        tracking = "databricks"

    print(f"Project root: {ROOT}")
    print(f"MLflow tracking: {tracking} -> {os.getenv('MLFLOW_EXPERIMENT_NAME', 'predictive_maintenance')}")
    print(_cuda_info())
    print(
        f"Mode: {'fast' if args.fast else 'full'} | skip_lstm={skip_lstm} | "
        f"gbm_max_rows={gbm_max or 'default'} | datasets={args.datasets}",
        flush=True,
    )

    raw = ROOT / "data" / "raw" / "cmapss"
    upload_dir = args.upload_dir or os.getenv("CMAPSS_UPLOAD_DIR")
    from src.ingestion.cmapss_download import ensure_cmapss_raw

    ensure_cmapss_raw(
        raw,
        upload_dir=upload_dir,
        download=args.download or os.getenv("CMAPSS_ALLOW_DOWNLOAD", "").lower()
        in ("1", "true", "yes"),
    )

    if not args.skip_build:
        for ds in args.datasets:
            print(f"\n=== Phase 2 build {ds} ===", flush=True)
            build_cmapss_dataset(ds)

    run_phase3_all(
        args.datasets,
        lstm_epochs=lstm_epochs,
        skip_lstm=skip_lstm,
        gbm_max_rows=gbm_max,
        anomaly_max_rows=anomaly_max,
    )

    print("\n=== Verification ===", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "report_cmapss_mlflow.py")],
        cwd=str(ROOT),
        check=False,
    )
    if os.getenv("MLFLOW_TRACKING_URI") == "databricks":
        print(f"\nMLflow runs: Databricks workspace {os.getenv('DATABRICKS_HOST')}")
        print("  UI: Machine Learning → Experiments →", os.getenv("MLFLOW_EXPERIMENT_NAME"))
    else:
        print(f"\nMLflow runs: {ROOT / 'mlruns'}")
    print("Download models/, artifacts/, data/processed/*_predictions.parquet (mlruns/ only if local)")


if __name__ == "__main__":
    main()
