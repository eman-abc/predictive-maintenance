#!/usr/bin/env python
"""
Upload on-disk CMAPSS .pkl / .pt files to a Databricks MLflow run (raw artifacts).

Use when you only need files in Experiments → Artifacts (not Unity Catalog registry).

  python scripts/upload_cmapss_artifacts_to_mlflow.py --models-dir models
  python scripts/upload_cmapss_artifacts_to_mlflow.py --models-dir /content/predictive-maintenance/models

Requires MLFLOW_TRACKING_URI=databricks and DATABRICKS_HOST + DATABRICKS_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def _collect_files(models_dir: Path) -> list[Path]:
    exts = {".pkl", ".pt", ".json"}
    files: list[Path] = []
    if not models_dir.is_dir():
        return files
    for path in sorted(models_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload CMAPSS model files to MLflow run artifacts")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Folder containing rul_*.pkl, failure_*.pkl, etc.",
    )
    parser.add_argument(
        "--run-name",
        default="cmapss_pkl_bundle",
        help="MLflow run name (default: cmapss_pkl_bundle)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override MLFLOW_EXPERIMENT_NAME",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    files = _collect_files(models_dir)

    if not files:
        print(f"FAIL: no .pkl/.pt/.json files in {models_dir}", file=sys.stderr)
        print("Unzip cmapss_colab_outputs.zip and point --models-dir at the folder that contains the pickles.", file=sys.stderr)
        if models_dir.exists():
            sample = list(models_dir.iterdir())[:15]
            print(f"Directory exists; contents ({len(list(models_dir.iterdir()))} items):", file=sys.stderr)
            for p in sample:
                print(f"  {p.name}", file=sys.stderr)
        else:
            print("Directory does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"):
        print("Set DATABRICKS_HOST and DATABRICKS_TOKEN", file=sys.stderr)
        sys.exit(1)

    import mlflow

    os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    exp = args.experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/predictive_maintenance")
    mlflow.set_experiment(exp)

    manifest = [{"name": p.name, "bytes": p.stat().st_size} for p in files]
    print(f"Uploading {len(files)} file(s) from {models_dir} → experiment {exp!r}", flush=True)
    for row in manifest:
        print(f"  {row['name']} ({row['bytes']:,} bytes)", flush=True)

    with mlflow.start_run(run_name=args.run_name) as run:
        for path in files:
            mlflow.log_artifact(str(path), artifact_path="models")
            print(f"  logged {path.name}", flush=True)
        manifest_path = models_dir / "_upload_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            mlflow.log_artifact(str(manifest_path), artifact_path="models")
        finally:
            manifest_path.unlink(missing_ok=True)
        run_id = run.info.run_id

    print(f"\nDone. run_id={run_id}", flush=True)
    print("Databricks UI: Experiments → this run → Artifacts tab → models/", flush=True)


if __name__ == "__main__":
    main()
