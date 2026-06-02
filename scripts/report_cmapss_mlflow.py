#!/usr/bin/env python
"""
Print CMAPSS Phase 3 training coverage for supervisor review.

Checks:
  - artifacts/cmapss_training_registry.json (local index)
  - MLflow experiment runs tagged pipeline=cmapss_phase3
  - Parent runs FD00X_phase3_summary (winner + test metrics + Cox test metrics)
  - Nested runs FD00X_rul_cox (when Cox trained)
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


def _run_name(row) -> str:
    return str(row.get("tags.mlflow.runName") or row.get("run_name", ""))


def _search_mlflow_runs() -> "pd.DataFrame":
    import pandas as pd

    try:
        import mlflow
        from mlflow.entities import ViewType
    except ImportError:
        return pd.DataFrame()

    uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance")
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return pd.DataFrame()

    return mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=500,
    )


def _mlflow_summary_runs(runs) -> list[dict]:
    if runs is None or runs.empty:
        return []

    out = []
    for _, row in runs.iterrows():
        name = _run_name(row)
        if not name.endswith("_phase3_summary"):
            continue
        dataset_id = row.get("params.dataset_id") or name.replace("_phase3_summary", "")
        skip_cox = row.get("params.skip_cox")
        pipeline_tag = row.get("tags.pipeline")
        out.append(
            {
                "run_name": name,
                "run_id": row.get("run_id"),
                "dataset_id": dataset_id,
                "pipeline_tagged": pipeline_tag == "cmapss_phase3",
                "winner": row.get("params.winner"),
                "skip_cox": str(skip_cox).lower() in ("true", "1"),
                "test_nasa": row.get("metrics.test_rul_score") or row.get("metrics.test_nasa_score"),
                "test_rmse": row.get("metrics.test_rmse"),
                "test_cox_nasa": row.get("metrics.test_cox_rul_score"),
                "test_cox_rmse": row.get("metrics.test_cox_rmse"),
                "start_time": str(row.get("start_time", "")),
                "training_batch": row.get("params.training_batch")
                or row.get("tags.training_batch"),
            }
        )
    return sorted(out, key=lambda r: str(r.get("dataset_id", "")))


def _latest_summary_per_dataset(runs: list[dict]) -> dict[str, dict]:
    """Most recent parent run per dataset (for verification table)."""
    by_ds: dict[str, list[dict]] = {}
    for row in runs:
        ds = str(row.get("dataset_id", ""))
        by_ds.setdefault(ds, []).append(row)
    latest: dict[str, dict] = {}
    for ds, items in by_ds.items():
        latest[ds] = max(items, key=lambda r: str(r.get("start_time", "")))
    return latest


def _summary_run_counts(runs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in runs:
        ds = str(row.get("dataset_id", ""))
        counts[ds] = counts.get(ds, 0) + 1
    return counts


def _mlflow_cox_nested_runs(runs) -> dict[str, dict]:
    """Map dataset_id -> nested Cox run metadata."""
    if runs is None or runs.empty:
        return {}

    by_ds: dict[str, dict] = {}
    for _, row in runs.iterrows():
        name = _run_name(row)
        if not name.endswith("_rul_cox"):
            continue
        dataset_id = name.replace("_rul_cox", "")
        by_ds[dataset_id] = {
            "run_name": name,
            "run_id": row.get("run_id"),
            "val_cox_nasa": row.get("metrics.val_rul_score"),
            "val_cox_rmse": row.get("metrics.val_rmse"),
            "val_concordance": row.get("metrics.val_concordance"),
            "n_cox_features": row.get("params.n_cox_features"),
            "cox_error": row.get("params.cox_error"),
        }
    return by_ds


def build_report() -> dict:
    registry = _load_registry()
    runs = _search_mlflow_runs()
    mlflow_runs = _mlflow_summary_runs(runs)
    latest_mlflow = _latest_summary_per_dataset(mlflow_runs)
    run_counts = _summary_run_counts(mlflow_runs)
    cox_runs = _mlflow_cox_nested_runs(runs)
    by_ds: dict[str, dict] = {}

    for ds in CMAPSS_DATASET_IDS:
        pred = PROCESSED_DIR / f"cmapss_{ds}_predictions.parquet"
        summary = ROOT / "artifacts" / f"cmapss_{ds}_phase3_summary.json"
        reg_entry = (registry.get("datasets") or {}).get(ds, {})
        mlflow_match = [latest_mlflow[ds]] if ds in latest_mlflow else []
        cox_entry = cox_runs.get(ds, {})
        summary_data = {}
        if summary.exists():
            summary_data = json.loads(summary.read_text(encoding="utf-8"))

        skip_cox = (
            reg_entry.get("skip_cox")
            if "skip_cox" in reg_entry
            else (mlflow_match[0].get("skip_cox") if mlflow_match else summary_data.get("skip_cox"))
        )
        test_cox_nasa = (
            reg_entry.get("test_cox_nasa_score")
            or (mlflow_match[0].get("test_cox_nasa") if mlflow_match else None)
            or (summary_data.get("cox_test_metrics") or {}).get("rul_score")
        )
        cox_nested = bool(cox_entry) and not cox_entry.get("cox_error")
        cox_logged = (
            not skip_cox
            and (
                cox_nested
                or test_cox_nasa is not None
                or (ROOT / "models" / f"survival_{ds}.pkl").exists()
            )
        )

        by_ds[ds] = {
            "predictions_parquet": pred.exists(),
            "summary_json": summary.exists(),
            "registry": bool(reg_entry),
            "mlflow_summary_run": len(mlflow_match) > 0,
            "mlflow_summary_run_count": run_counts.get(ds, 0),
            "training_batch": (
                mlflow_match[0].get("training_batch") if mlflow_match else reg_entry.get("training_batch")
            ),
            "mlflow_pipeline_tagged": any(
                m.get("pipeline_tagged") for m in mlflow_match
            ),
            "mlflow_run_id": (
                mlflow_match[0].get("run_id") if mlflow_match else reg_entry.get("mlflow_run_id")
            ),
            "mlflow_cox_nested_run": cox_nested,
            "cox_logged": cox_logged,
            "skip_cox": bool(skip_cox),
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
            "test_cox_nasa_score": test_cox_nasa,
            "test_cox_rmse": (
                reg_entry.get("test_cox_rmse")
                or (mlflow_match[0].get("test_cox_rmse") if mlflow_match else None)
                or (summary_data.get("cox_test_metrics") or {}).get("rmse")
            ),
        }

    complete = [
        ds
        for ds, s in by_ds.items()
        if s["predictions_parquet"] and s["summary_json"] and s["mlflow_summary_run"]
    ]
    cox_complete = [
        ds for ds, s in by_ds.items() if s["cox_logged"] and not s.get("skip_cox")
    ]
    return {
        "experiment": os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance"),
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")),
        "registry_path": str(REGISTRY_PATH),
        "registry_exists": REGISTRY_PATH.exists(),
        "datasets_complete": complete,
        "cox_datasets_complete": cox_complete,
        "all_complete": len(complete) == len(CMAPSS_DATASET_IDS),
        "all_cox_complete": len(cox_complete) == len(CMAPSS_DATASET_IDS),
        "by_dataset": by_ds,
        "mlflow_runs": mlflow_runs,
        "mlflow_cox_runs": cox_runs,
    }


def _print_table(report: dict) -> None:
    print(f"Experiment: {report['experiment']}")
    print(f"MLflow URI:  {report['tracking_uri']}")
    print(f"Registry:    {report['registry_path']} ({'found' if report['registry_exists'] else 'missing'})")
    print()
    header = (
        f"{'Dataset':<8} {'Pred':^5} {'JSON':^5} {'MLflow':^7} "
        f"{'#Runs':^5} {'Tag':^4} {'Cox':^4} {'Winner':<6} {'NASA':>8} {'CoxNASA':>8}"
    )
    print(header)
    print("-" * len(header))
    for ds, s in report["by_dataset"].items():
        cox_cell = "skip" if s.get("skip_cox") else ("Y" if s.get("cox_logged") else "N")
        print(
            f"{ds:<8} "
            f"{'Y' if s['predictions_parquet'] else 'N':^5} "
            f"{'Y' if s['summary_json'] else 'N':^5} "
            f"{'Y' if s['mlflow_summary_run'] else 'N':^7} "
            f"{s.get('mlflow_summary_run_count', 0):^5} "
            f"{'Y' if s.get('mlflow_pipeline_tagged') else 'N':^4} "
            f"{cox_cell:^4} "
            f"{str(s.get('winner') or '-'):<6} "
            f"{_fmt_metric(s.get('test_nasa_score')):>8} "
            f"{_fmt_metric(s.get('test_cox_nasa_score')):>8}"
        )
    print()
    print("Table uses the latest MLflow parent run per dataset. #Runs = total parent runs kept in the experiment.")
    batches = sorted(
        {s.get("training_batch") for s in report["by_dataset"].values() if s.get("training_batch")}
    )
    if batches:
        print(f"Latest training_batch tags: {', '.join(batches)}")
    print()
    if report["all_complete"]:
        print("All four CMAPSS subsets are trained and logged to MLflow.")
    else:
        missing = [d for d in CMAPSS_DATASET_IDS if d not in report["datasets_complete"]]
        print(f"Incomplete subsets: {', '.join(missing)}")
        print("Train all: python scripts/train_cmapss_phase3.py --all")
    if report.get("all_cox_complete"):
        print("Cox PH logged for all four subsets (nested FD00X_rul_cox + test_cox_* metrics).")
    elif not report.get("all_cox_complete"):
        missing_cox = [
            d
            for d in CMAPSS_DATASET_IDS
            if d not in report.get("cox_datasets_complete", [])
        ]
        print(f"Cox not fully logged: {', '.join(missing_cox)} (re-train without --skip-cox)")
    print()
    print("Open MLflow UI: mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000")
    print("Filter runs: tags.pipeline = 'cmapss_phase3'")
    print("Compare re-trains: tags.training_batch = '<your label>' or sort by Start time")
    print("Nested Cox runs: FD00X_rul_cox under each FD00X_phase3_summary parent")


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
