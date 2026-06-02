#!/usr/bin/env python
"""
Print CMAPSS Phase 3 training coverage for supervisor review.

Checks:
  - artifacts/cmapss_training_registry.json (local index)
  - MLflow experiment runs tagged pipeline=cmapss_phase3
  - Processed predictions parquet per FD subset

Usage:
  python scripts/report_cmapss_mlflow.py
  python scripts/report_cmapss_mlflow.py --json
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

from src.ingestion.cmapss_config import CMAPSS_DATASET_IDS  # noqa: E402

REGISTRY_PATH = ROOT / "artifacts" / "cmapss_training_registry.json"
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", str(ROOT / "data" / "processed")))


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"datasets": {}}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _mlflow_summary_runs() -> list[dict]:
    try:
        import mlflow
        from mlflow.entities import ViewType
    except ImportError:
        return []

    uri = os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance")
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=200,
    )
    if runs.empty:
        return []

    out = []
    for _, row in runs.iterrows():
        name = str(row.get("tags.mlflow.runName") or row.get("run_name", ""))
        if not name.endswith("_phase3_summary"):
            continue
        dataset_id = row.get("params.dataset_id") or name.replace("_phase3_summary", "")
        pipeline_tag = row.get("tags.pipeline")
        out.append(
            {
                "run_name": name,
                "run_id": row.get("run_id"),
                "dataset_id": dataset_id,
                "pipeline_tagged": pipeline_tag == "cmapss_phase3",
                "winner": row.get("params.winner"),
                "test_nasa": row.get("metrics.test_rul_score") or row.get("metrics.test_nasa_score"),
                "test_rmse": row.get("metrics.test_rmse"),
                "start_time": str(row.get("start_time", "")),
            }
        )
    return sorted(out, key=lambda r: str(r.get("dataset_id", "")))


def build_report() -> dict:
    registry = _load_registry()
    mlflow_runs = _mlflow_summary_runs()
    by_ds: dict[str, dict] = {}

    for ds in CMAPSS_DATASET_IDS:
        pred = PROCESSED_DIR / f"cmapss_{ds}_predictions.parquet"
        summary = ROOT / "artifacts" / f"cmapss_{ds}_phase3_summary.json"
        reg_entry = (registry.get("datasets") or {}).get(ds, {})
        mlflow_match = [r for r in mlflow_runs if r.get("dataset_id") == ds]
        summary_data = {}
        if summary.exists():
            summary_data = json.loads(summary.read_text(encoding="utf-8"))

        by_ds[ds] = {
            "predictions_parquet": pred.exists(),
            "summary_json": summary.exists(),
            "registry": bool(reg_entry),
            "mlflow_summary_run": len(mlflow_match) > 0,
            "mlflow_pipeline_tagged": any(
                m.get("pipeline_tagged") for m in mlflow_match
            ),
            "mlflow_run_id": (
                mlflow_match[0].get("run_id") if mlflow_match else reg_entry.get("mlflow_run_id")
            ),
            "winner": (
                reg_entry.get("winner")
                or (mlflow_match[0].get("winner") if mlflow_match else None)
                or summary_data.get("winner")
            ),
            "test_nasa_score": (
                reg_entry.get("test_nasa_score")
                or (mlflow_match[0].get("test_nasa") if mlflow_match else None)
                or (summary_data.get("test_metrics") or {}).get("rul_score")
            ),
            "test_rmse": (
                reg_entry.get("test_rmse")
                or (mlflow_match[0].get("test_rmse") if mlflow_match else None)
                or (summary_data.get("test_metrics") or {}).get("rmse")
            ),
        }

    complete = [
        ds
        for ds, s in by_ds.items()
        if s["predictions_parquet"] and s["summary_json"] and s["mlflow_summary_run"]
    ]
    return {
        "experiment": os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance"),
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")),
        "registry_path": str(REGISTRY_PATH),
        "registry_exists": REGISTRY_PATH.exists(),
        "datasets_complete": complete,
        "all_complete": len(complete) == len(CMAPSS_DATASET_IDS),
        "by_dataset": by_ds,
        "mlflow_runs": mlflow_runs,
    }


def _print_table(report: dict) -> None:
    print(f"Experiment: {report['experiment']}")
    print(f"MLflow URI:  {report['tracking_uri']}")
    print(f"Registry:    {report['registry_path']} ({'found' if report['registry_exists'] else 'missing'})")
    print()
    header = (
        f"{'Dataset':<8} {'Pred':^5} {'JSON':^5} {'MLflow':^7} "
        f"{'Tag':^4} {'Winner':<6} {'NASA':>8} {'RMSE':>8}"
    )
    print(header)
    print("-" * len(header))
    for ds, s in report["by_dataset"].items():
        print(
            f"{ds:<8} "
            f"{'Y' if s['predictions_parquet'] else 'N':^5} "
            f"{'Y' if s['summary_json'] else 'N':^5} "
            f"{'Y' if s['mlflow_summary_run'] else 'N':^7} "
            f"{'Y' if s.get('mlflow_pipeline_tagged') else 'N':^4} "
            f"{str(s.get('winner') or '-'):<6} "
            f"{_fmt_metric(s.get('test_nasa_score')):>8} "
            f"{_fmt_metric(s.get('test_rmse')):>8}"
        )
    print()
    if report["all_complete"]:
        print("All four CMAPSS subsets are trained and logged.")
    else:
        missing = [d for d in CMAPSS_DATASET_IDS if d not in report["datasets_complete"]]
        print(f"Incomplete subsets: {', '.join(missing)}")
        print("Train all: python scripts/train_cmapss_phase3.py --all")
    print()
    print("Open MLflow UI: mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000")
    print("Filter runs: tags.pipeline = 'cmapss_phase3' (Tag=Y after re-train with current code)")


def _fmt_metric(value) -> str:
    if value is None:
        return "-"
    try:
        fv = float(value)
        if fv != fv:  # NaN
            return "-"
        return f"{fv:.2f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS training verification report")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_table(report)
    sys.exit(0 if report["all_complete"] else 1)


if __name__ == "__main__":
    main()
