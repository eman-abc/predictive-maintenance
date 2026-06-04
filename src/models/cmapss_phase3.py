"""Phase 3: train, compare, evaluate, and export CMAPSS predictions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.alerts.alert_payload import snapshot_sensor_readings
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.alerts.threshold_engine import ThresholdEngine
from src.models.cmapss_eval import (
    evaluate_failure_classification,
    evaluate_rul,
    last_cycle_per_unit,
    load_feature_columns,
    prepare_xy,
    prepare_xy_validation,
    rank_models,
    rul_validation_frame,
)
from src.models.cmapss_splits import filter_by_units, split_unit_ids
from src.models.anomaly_detector import (
    AnomalyDetector,
    evaluate_anomaly_degradation_proxy,
)
from src.models.failure_classifier import FailureClassifier
from src.models.lstm_model import LSTMModel
from src.ingestion.cmapss_config import CMAPSS_DATASET_IDS
from src.models.rul_regressor import RULRegressor
from src.models.cmapss_survival import (
    add_survival_columns,
    evaluate_cox_rul,
    select_cox_features,
)
from src.models.mlflow_registry import register_phase3_models
from src.models.survival_model import SurvivalModel

load_dotenv()

MODELS_DIR = Path("models")
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed"))
ARTIFACTS_DIR = Path("artifacts")
REGISTRY_PATH = ARTIFACTS_DIR / "cmapss_training_registry.json"
MLFLOW_PIPELINE_TAG = "cmapss_phase3"
PIPELINE_VERSION = "phase3_v2"
LSTM_WINDOW = 30
VAL_FRACTION = 0.2
# GBM on full FD002/FD004 row counts can take hours; subsample for fit only.
GBM_MAX_TRAIN_ROWS = 250_000
# UC5 failure windows (cycles as proxy for 24h / 72h planning horizons)
FAILURE_HORIZONS = (30, 72)
ANOMALY_MIN_RUL_FIT = 30
ANOMALY_MAX_TRAIN_ROWS = 100_000
COX_MAX_TRAIN_ROWS = 80_000
COX_MAX_FEATURES = 40
RUL_WINNER_CANDIDATES = ("rf", "gbm", "lstm")


def _subsample_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def _training_batch_id(explicit: str | None = None) -> str:
    """Stable label for one training session (re-runs get a new batch; MLflow keeps all runs)."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    env = os.getenv("MLFLOW_TRAINING_BATCH_ID", "").strip()
    if env:
        return env
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _mlflow_setup(dataset_id: str) -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(uri)
    name = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance")
    mlflow.set_experiment(name)
    return name


def _registry_entry(summary: dict[str, Any], *, mlflow_run_id: str | None) -> dict[str, Any]:
    test_m = summary.get("test_metrics") or {}
    cox_test = summary.get("cox_test_metrics") or {}
    return {
        "dataset_id": summary["dataset_id"],
        "winner": summary.get("winner"),
        "test_rmse": test_m.get("rmse"),
        "test_nasa_score": test_m.get("rul_score"),
        "skip_cox": bool(summary.get("skip_cox")),
        "test_cox_rmse": cox_test.get("rmse"),
        "test_cox_nasa_score": cox_test.get("rul_score"),
        "predictions_path": summary.get("predictions_path"),
        "summary_json": str(ARTIFACTS_DIR / f"cmapss_{summary['dataset_id']}_phase3_summary.json"),
        "survival_model": str(MODELS_DIR / f"survival_{summary['dataset_id']}.pkl"),
        "registered_models": summary.get("registered_models") or {},
        "mlflow_experiment": os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance"),
        "mlflow_experiment_id": summary.get("mlflow_experiment_id")
        or os.getenv("MLFLOW_EXPERIMENT_ID")
        or None,
        "mlflow_run_name": f"{summary['dataset_id']}_phase3_summary",
        "mlflow_run_id": mlflow_run_id,
        "training_batch": summary.get("training_batch"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def update_training_registry(summary: dict[str, Any], *, mlflow_run_id: str | None = None) -> Path:
    """Merge one dataset's Phase 3 summary into artifacts/cmapss_training_registry.json."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    entry = _registry_entry(summary, mlflow_run_id=mlflow_run_id)
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = {
            "pipeline": MLFLOW_PIPELINE_TAG,
            "pipeline_version": PIPELINE_VERSION,
            "experiment": entry["mlflow_experiment"],
            "datasets": {},
        }
    registry["updated_at"] = entry["trained_at"]
    exp_id = entry.get("mlflow_experiment_id") or os.getenv("MLFLOW_EXPERIMENT_ID")
    if exp_id:
        registry["mlflow_experiment_id"] = str(exp_id)
    prev = registry["datasets"].get(summary["dataset_id"], {})
    history = list(prev.get("mlflow_run_history") or [])
    if prev.get("mlflow_run_id"):
        history.append(
            {
                "mlflow_run_id": prev.get("mlflow_run_id"),
                "training_batch": prev.get("training_batch"),
                "trained_at": prev.get("trained_at"),
                "winner": prev.get("winner"),
                "test_nasa_score": prev.get("test_nasa_score"),
            }
        )
    entry["mlflow_run_history"] = history[-20:]
    registry["datasets"][summary["dataset_id"]] = entry
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return REGISTRY_PATH


def run_phase3_all(
    dataset_ids: list[str] | tuple[str, ...] | None = None,
    *,
    val_fraction: float = VAL_FRACTION,
    lstm_epochs: int = 15,
    skip_lstm: bool = False,
    skip_cox: bool = False,
    gbm_max_rows: int | None = None,
    anomaly_max_rows: int | None = None,
    training_batch: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate all requested CMAPSS subsets; update the training registry."""
    ids = list(dataset_ids or CMAPSS_DATASET_IDS)
    batch_id = _training_batch_id(training_batch)
    print(
        f"MLflow training_batch={batch_id!r} — each train creates NEW runs; "
        "previous experiment runs are not deleted.",
        flush=True,
    )
    summaries: dict[str, Any] = {}
    for dataset_id in ids:
        print(f"\n========== CMAPSS Phase 3: {dataset_id} ==========", flush=True)
        summaries[dataset_id] = run_phase3(
            dataset_id,
            val_fraction=val_fraction,
            lstm_epochs=lstm_epochs,
            skip_lstm=skip_lstm,
            skip_cox=skip_cox,
            gbm_max_rows=gbm_max_rows,
            anomaly_max_rows=anomaly_max_rows,
            training_batch=batch_id,
        )
    return summaries


def _train_rul(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    gbm_max_rows: int = GBM_MAX_TRAIN_ROWS,
) -> tuple[RULRegressor, dict[str, float], dict[str, float]]:
    fit_df = (
        _subsample_rows(train_df, gbm_max_rows)
        if model_type == "gbm"
        else train_df
    )
    X_train = fit_df[feature_cols].fillna(0)
    y_train = fit_df["rul"]
    reg = RULRegressor(model_type=model_type)
    reg.feature_cols = feature_cols
    reg.model.fit(X_train, y_train)

    X_val, y_val = prepare_xy_validation(val_df, feature_cols)
    val_metrics = evaluate_rul(y_val.values, reg.predict(X_val))
    return reg, val_metrics, {}


def _train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    epochs: int = 15,
) -> tuple[LSTMModel, dict[str, float], dict[str, float]]:
    lstm = LSTMModel(input_size=len(feature_cols), hidden_size=64, num_layers=2)
    lstm.sequence_length = LSTM_WINDOW
    train_metrics = lstm.fit(
        train_df.sort_values(["unit_id", "cycle"]),
        feature_cols,
        epochs=epochs,
        batch_size=64,
    )

    y_val, val_preds = lstm.predict_validation(val_df, feature_cols)
    val_metrics = evaluate_rul(y_val, val_preds)
    return lstm, val_metrics, train_metrics


def _fit_failure_classifier(
    clf: FailureClassifier,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    *,
    gbm_max_rows: int = GBM_MAX_TRAIN_ROWS,
) -> None:
    """Fit GBM with balanced sample weights for imbalanced failure horizons."""
    from sklearn.utils.class_weight import compute_sample_weight

    fit_df = _subsample_rows(train_df, gbm_max_rows)
    X = fit_df[feature_cols].fillna(0)
    y = fit_df[label_col]
    weights = compute_sample_weight(class_weight="balanced", y=y)
    clf.feature_cols = feature_cols
    clf.model.fit(X, y, sample_weight=weights)


def _eval_failure_classifier(
    clf: FailureClassifier,
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    *,
    eval_protocol: str,
) -> dict[str, float | int | str]:
    X = df[feature_cols].fillna(0)
    y_true = df[label_col].values
    preds = clf.predict(X)
    proba = clf.predict_proba(X)
    return evaluate_failure_classification(
        y_true, preds, proba, eval_protocol=eval_protocol
    )


def _train_failure_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "failure_30",
    *,
    gbm_max_rows: int = GBM_MAX_TRAIN_ROWS,
) -> tuple[FailureClassifier, dict[str, float | int | str]]:
    """
    Train failure-within-horizon classifier (UC5 Component B).

    Validation uses non-terminal cycles on held-out engines (mixed 0/1 labels).
    """
    clf = FailureClassifier(model_type="gbm")
    _fit_failure_classifier(clf, train_df, feature_cols, label_col, gbm_max_rows=gbm_max_rows)

    val_eval = rul_validation_frame(val_df)
    metrics = _eval_failure_classifier(
        clf,
        val_eval,
        feature_cols,
        label_col,
        eval_protocol="val_non_terminal_cycles",
    )
    return clf, metrics


def _predict_rul_last_cycle(
    df: pd.DataFrame,
    rul_model: RULRegressor | LSTMModel,
    feature_cols: list[str],
) -> np.ndarray:
    sorted_df = df.sort_values(["unit_id", "cycle"])
    if isinstance(rul_model, LSTMModel):
        return rul_model.predict(sorted_df, feature_cols)
    last = last_cycle_per_unit(sorted_df)
    return rul_model.predict(last[feature_cols].fillna(0))


def _retrain_rul_winner(
    winner: str,
    train_full: pd.DataFrame,
    feature_cols: list[str],
    *,
    lstm_epochs: int,
    gbm_max_rows: int = GBM_MAX_TRAIN_ROWS,
) -> RULRegressor | LSTMModel:
    if winner == "lstm":
        model = LSTMModel(input_size=len(feature_cols), hidden_size=64, num_layers=2)
        model.sequence_length = LSTM_WINDOW
        model.fit(
            train_full.sort_values(["unit_id", "cycle"]),
            feature_cols,
            epochs=lstm_epochs,
        )
        return model
    reg = RULRegressor(model_type=winner)
    reg.feature_cols = feature_cols
    fit_df = (
        _subsample_rows(train_full, gbm_max_rows)
        if winner == "gbm"
        else train_full
    )
    reg.model.fit(fit_df[feature_cols].fillna(0), fit_df["rul"])
    return reg


def _train_cox(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    max_rows: int = COX_MAX_TRAIN_ROWS,
    max_features: int = COX_MAX_FEATURES,
) -> tuple[SurvivalModel | None, dict[str, float]]:
    """
    Cox PH survival baseline (lifelines): median remaining life + survival probabilities.

    Does not replace the RUL regression winner; logged alongside RF/GBM/LSTM.
    """
    try:
        cox_features = select_cox_features(train_df, feature_cols, max_features=max_features)
        train_surv = add_survival_columns(train_df, is_train=True)
        cox = SurvivalModel()
        fit_metrics = cox.fit(
            train_surv,
            cox_features,
            max_rows=max_rows,
        )

        val_last = last_cycle_per_unit(add_survival_columns(val_df, is_train=True))
        rul_cox = cox.predict_remaining_rul(val_last, val_last["cycle"].values)
        val_metrics = evaluate_cox_rul(
            val_last["rul"].values,
            rul_cox,
            concordance=fit_metrics.get("concordance"),
        )
        return cox, val_metrics
    except Exception as exc:
        print(f"  Cox PH skipped ({exc})", flush=True)
        return None, {"error": str(exc)}


def _train_anomaly_detector(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    anomaly_max_rows: int = ANOMALY_MAX_TRAIN_ROWS,
) -> tuple[AnomalyDetector, dict[str, float | int | str]]:
    detector = AnomalyDetector()
    fit_info = detector.fit(
        train_df,
        feature_cols,
        min_rul=ANOMALY_MIN_RUL_FIT,
        max_rows=anomaly_max_rows,
    )
    val_eval = rul_validation_frame(val_df)
    val_scores, _ = detector.predict_scores(val_eval)
    metrics = evaluate_anomaly_degradation_proxy(
        val_eval,
        val_scores,
        rul_threshold=ANOMALY_MIN_RUL_FIT,
        eval_protocol="val_non_terminal_cycles",
    )
    metrics.update({f"fit_{k}": v for k, v in fit_info.items()})
    return detector, metrics


def build_fleet_predictions(
    test_df: pd.DataFrame,
    rul_model: RULRegressor | LSTMModel,
    failure_clf_30: FailureClassifier,
    feature_cols: list[str],
    *,
    model_name: str,
    failure_clf_72: FailureClassifier | None = None,
    anomaly_detector: AnomalyDetector | None = None,
    survival_model: SurvivalModel | None = None,
    dataset_id: str = "",
) -> pd.DataFrame:
    """Last-cycle predictions + alerts for each test engine."""
    test_sorted = test_df.sort_values(["unit_id", "cycle"])
    last = last_cycle_per_unit(test_sorted).reset_index(drop=True)
    rul_pred = _predict_rul_last_cycle(test_df, rul_model, feature_cols)
    X_last = last[feature_cols].fillna(0)
    failure_prob_30 = failure_clf_30.predict_proba(X_last)
    failure_prob_72 = (
        failure_clf_72.predict_proba(X_last)
        if failure_clf_72 is not None
        else np.zeros(len(last))
    )
    if anomaly_detector is not None:
        anomaly_scores, anomaly_flags = anomaly_detector.predict_scores(last)
    else:
        anomaly_scores = np.zeros(len(last))
        anomaly_flags = np.zeros(len(last), dtype=int)

    if survival_model is not None:
        X_surv = last[survival_model.feature_cols].fillna(0)
        cycles = last["cycle"].values
        rul_cox = survival_model.predict_remaining_rul(X_surv, cycles)
        surv_prob_30 = survival_model.predict_survival_probability(
            X_surv, cycles, horizon=30
        )
        surv_prob_72 = survival_model.predict_survival_probability(
            X_surv, cycles, horizon=72
        )
    else:
        rul_cox = np.full(len(last), np.nan)
        surv_prob_30 = np.full(len(last), np.nan)
        surv_prob_72 = np.full(len(last), np.nan)

    engine = ThresholdEngine()

    records = []
    for pos in range(len(last)):
        unit_id = int(last.loc[pos, "unit_id"])
        rul_p = float(rul_pred[pos])
        fail_30 = float(failure_prob_30[pos])
        fail_72 = float(failure_prob_72[pos])
        anom = float(anomaly_scores[pos])
        sensor_snap = snapshot_sensor_readings(last.loc[pos])
        assessment = engine.assess(
            f"ENG-{unit_id:03d}",
            rul_p,
            fail_30,
            anomaly_score=anom,
            sensor_readings=sensor_snap,
        )
        level = assessment.alert_level.value
        escalation = (
            "L2-Critical"
            if level == "critical"
            else "L1-Warning"
            if level == "warning"
            else "L0-Normal"
        )
        records.append(
            {
                "dataset_id": dataset_id,
                "unit_id": unit_id,
                "asset_id": f"ENG-{unit_id:03d}",
                "cycle": int(last.loc[pos, "cycle"]),
                "rul_true": float(last.loc[pos, "rul"]),
                "rul_pred": rul_p,
                "rul_pred_cox": float(rul_cox[pos]),
                "survival_prob_30": float(surv_prob_30[pos]),
                "survival_prob_72": float(surv_prob_72[pos]),
                "time_to_failure_cycles": assessment.time_to_failure_cycles,
                "failure_prob_30": fail_30,
                "failure_prob_72": fail_72,
                "failure_prob": fail_30,
                "failure_30_true": int(last.loc[pos, "failure_30"]),
                "failure_72_true": int(last.loc[pos, "failure_72"]),
                "anomaly_score": anom,
                "is_anomaly": int(anomaly_flags[pos]),
                "risk_score": assessment.risk_score,
                "health_score": assessment.health_score,
                "alert_level": level,
                "alert_message": assessment.message,
                "recommended_action": assessment.recommended_action,
                "sensor_readings_json": json.dumps(sensor_snap),
                "escalation_tier": escalation,
                "rul_model": model_name,
            }
        )

    return pd.DataFrame(records)


def run_phase3(
    dataset_id: str = "FD001",
    *,
    val_fraction: float = VAL_FRACTION,
    lstm_epochs: int = 15,
    skip_lstm: bool = False,
    skip_cox: bool = False,
    gbm_max_rows: int | None = None,
    anomaly_max_rows: int | None = None,
    training_batch: str | None = None,
) -> dict[str, Any]:
    """
    Full Phase 3 pipeline: engine split, model comparison, test eval, fleet export.

    Each call creates a **new** MLflow parent run (and nested runs). Re-training does not
    delete or overwrite prior runs in the tracking store. Local ``models/`` and fleet
    Parquet are overwritten with the latest artifacts only.

    Fast-training options (still logged to MLflow):
    - ``skip_lstm``: compare RF vs GBM only (CPU-friendly on Colab).
    - ``skip_cox``: skip lifelines Cox PH (faster runs).
    - ``gbm_max_rows`` / ``anomaly_max_rows``: cap rows used for GBM / Isolation Forest fit.
    """
    gbm_cap = gbm_max_rows if gbm_max_rows is not None else GBM_MAX_TRAIN_ROWS
    anomaly_cap = anomaly_max_rows if anomaly_max_rows is not None else ANOMALY_MAX_TRAIN_ROWS
    batch_id = _training_batch_id(training_batch)
    source = os.getenv("CMAPSS_SOURCE", "local")

    _mlflow_setup(dataset_id)
    MODELS_DIR.mkdir(exist_ok=True)

    train_path = PROCESSED_DIR / f"cmapss_{dataset_id}_train.parquet"
    test_path = PROCESSED_DIR / f"cmapss_{dataset_id}_test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed data missing for {dataset_id}. "
            "Run: python scripts/build_cmapss_dataset.py --dataset " + dataset_id
        )

    train_full = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    feature_cols = load_feature_columns(ARTIFACTS_DIR, dataset_id)

    train_units, val_units = split_unit_ids(
        train_full["unit_id"].unique(), val_fraction=val_fraction
    )
    train_df = filter_by_units(train_full, train_units)
    val_df = filter_by_units(train_full, val_units)

    val_results: dict[str, dict[str, float]] = {}

    with mlflow.start_run(run_name=f"{dataset_id}_phase3_summary"):
        mlflow.set_tags(
            {
                "pipeline": MLFLOW_PIPELINE_TAG,
                "pipeline_version": PIPELINE_VERSION,
                "dataset_id": dataset_id,
                "task": "cmapss_rul_failure_anomaly",
                "training_batch": batch_id,
                "source": source,
            }
        )
        mlflow.log_param("dataset_id", dataset_id)
        mlflow.log_param("training_batch", batch_id)
        mlflow.log_param("source", source)
        mlflow.log_param("val_fraction", val_fraction)
        mlflow.log_param("lstm_window", LSTM_WINDOW)
        mlflow.log_param("skip_lstm", skip_lstm)
        mlflow.log_param("skip_cox", skip_cox)
        mlflow.log_param("cox_max_train_rows", COX_MAX_TRAIN_ROWS)
        mlflow.log_param("cox_max_features", COX_MAX_FEATURES)
        mlflow.log_param("gbm_max_train_rows", gbm_cap)
        mlflow.log_param("anomaly_max_train_rows", anomaly_cap)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_units", len(train_units))
        mlflow.log_param("val_units", len(val_units))

        for model_type in ("rf", "gbm"):
            print(f"[{dataset_id}] Training RUL {model_type}...", flush=True)
            with mlflow.start_run(run_name=f"{dataset_id}_rul_{model_type}", nested=True):
                reg, val_m, _ = _train_rul(
                    model_type, train_df, val_df, feature_cols, gbm_max_rows=gbm_cap
                )
                mlflow.log_param("model_type", model_type)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
                val_results[model_type] = val_m

        if not skip_lstm:
            print(f"[{dataset_id}] Training RUL lstm ({lstm_epochs} epochs)...", flush=True)
            with mlflow.start_run(run_name=f"{dataset_id}_rul_lstm", nested=True):
                lstm, val_m, train_m = _train_lstm(
                    train_df, val_df, feature_cols, epochs=lstm_epochs
                )
                mlflow.log_param("model_type", "lstm")
                mlflow.log_param("sequence_length", LSTM_WINDOW)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
                mlflow.log_metric("train_final_loss", train_m.get("final_loss", 0))
                val_results["lstm"] = val_m
        else:
            mlflow.log_param("lstm_skipped", True)
            print(f"[{dataset_id}] Skipping LSTM (skip_lstm=True)", flush=True)

        cox_model: SurvivalModel | None = None
        cox_val_metrics: dict[str, float] = {}
        if not skip_cox:
            print(f"[{dataset_id}] Training Cox PH survival (lifelines)...", flush=True)
            with mlflow.start_run(run_name=f"{dataset_id}_rul_cox", nested=True):
                cox_model, cox_val_metrics = _train_cox(
                    train_df, val_df, feature_cols, max_rows=COX_MAX_TRAIN_ROWS
                )
                mlflow.log_param("model_type", "cox_ph")
                mlflow.log_param("cox_max_train_rows", COX_MAX_TRAIN_ROWS)
                mlflow.log_param("cox_max_features", COX_MAX_FEATURES)
                if "error" not in cox_val_metrics:
                    if cox_model is not None:
                        mlflow.log_param("n_cox_features", len(cox_model.feature_cols))
                    for key, value in cox_val_metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"val_{key}", float(value))
                    val_results["cox"] = {
                        k: v for k, v in cox_val_metrics.items() if k in ("rmse", "rul_score", "mae")
                    }
                else:
                    mlflow.log_param("cox_error", cox_val_metrics["error"])
        else:
            mlflow.log_param("cox_skipped", True)

        rul_candidates = {
            k: v for k, v in val_results.items() if k in RUL_WINNER_CANDIDATES
        }
        winner = rank_models(rul_candidates)
        print(f"[{dataset_id}] Winner: {winner} (val NASA {val_results[winner]['rul_score']:.2f})", flush=True)
        mlflow.log_param("winner", winner)
        mlflow.log_metrics(
            {f"winner_val_{k}": v for k, v in val_results[winner].items()}
        )

        failure_clf_val_metrics: dict[str, dict[str, float | int | str]] = {}
        for horizon in FAILURE_HORIZONS:
            label_col = f"failure_{horizon}"
            print(f"[{dataset_id}] Training {label_col} classifier...", flush=True)
            with mlflow.start_run(
                run_name=f"{dataset_id}_{label_col}_gbm", nested=True
            ):
                clf, clf_val_m = _train_failure_classifier(
                    train_df,
                    val_df,
                    feature_cols,
                    label_col=label_col,
                    gbm_max_rows=gbm_cap,
                )
                failure_clf_val_metrics[label_col] = clf_val_m
                mlflow.log_param("label_col", label_col)
                mlflow.log_param("eval_protocol", clf_val_m["eval_protocol"])
                for key, value in clf_val_m.items():
                    if key == "eval_protocol":
                        continue
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"val_{key}", float(value))

        # Retrain winner + classifiers on all training engines before test scoring
        best_rul = _retrain_rul_winner(
            winner,
            train_full,
            feature_cols,
            lstm_epochs=lstm_epochs,
            gbm_max_rows=gbm_cap,
        )
        failure_clf = FailureClassifier(model_type="gbm")
        _fit_failure_classifier(
            failure_clf, train_full, feature_cols, "failure_30", gbm_max_rows=gbm_cap
        )
        failure_clf_72 = FailureClassifier(model_type="gbm")
        _fit_failure_classifier(
            failure_clf_72, train_full, feature_cols, "failure_72", gbm_max_rows=gbm_cap
        )

        if winner == "lstm":
            best_rul.save(MODELS_DIR / f"rul_lstm_{dataset_id}.pt")
        else:
            best_rul.save(MODELS_DIR / f"rul_{winner}_{dataset_id}.pkl")
        failure_clf.save(MODELS_DIR / f"failure_30_{dataset_id}.pkl")
        failure_clf_72.save(MODELS_DIR / f"failure_72_{dataset_id}.pkl")

        print(f"[{dataset_id}] Training anomaly detector (Isolation Forest)...", flush=True)
        with mlflow.start_run(run_name=f"{dataset_id}_anomaly_iforest", nested=True):
            anomaly_det, anomaly_val_m = _train_anomaly_detector(
                train_df, val_df, feature_cols, anomaly_max_rows=anomaly_cap
            )
            mlflow.log_param("min_rul_fit", ANOMALY_MIN_RUL_FIT)
            for key, value in anomaly_val_m.items():
                if key == "eval_protocol":
                    continue
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"val_{key}", float(value))
        anomaly_det.fit(
            train_full,
            feature_cols,
            min_rul=ANOMALY_MIN_RUL_FIT,
            max_rows=anomaly_cap,
        )
        anomaly_det.save(MODELS_DIR / f"anomaly_{dataset_id}.pkl")

        if cox_model is not None:
            train_surv = add_survival_columns(train_full, is_train=True)
            cox_model.fit(
                train_surv,
                cox_model.feature_cols,
                max_rows=COX_MAX_TRAIN_ROWS,
            )
            surv_path = MODELS_DIR / f"survival_{dataset_id}.pkl"
            cox_model.save(surv_path)
            mlflow.log_artifact(str(surv_path), artifact_path="models")
            print(f"[{dataset_id}] Saved Cox model", flush=True)

        test_last = last_cycle_per_unit(test_df)
        failure_clf_test_metrics = {
            "failure_30": _eval_failure_classifier(
                failure_clf,
                test_last,
                feature_cols,
                "failure_30",
                eval_protocol="test_last_cycle",
            ),
            "failure_72": _eval_failure_classifier(
                failure_clf_72,
                test_last,
                feature_cols,
                "failure_72",
                eval_protocol="test_last_cycle",
            ),
        }
        for label_col, m in failure_clf_test_metrics.items():
            if "roc_auc" in m:
                print(
                    f"[{dataset_id}] Test {label_col} ROC-AUC={m['roc_auc']:.3f} "
                    f"F1={m['f1']:.3f} (n_pos={m['n_positive']}, n_neg={m['n_negative']})",
                    flush=True,
                )

        test_anomaly_scores, test_anomaly_flags = anomaly_det.predict_scores(test_last)
        anomaly_test_metrics = evaluate_anomaly_degradation_proxy(
            test_last,
            test_anomaly_scores,
            rul_threshold=ANOMALY_MIN_RUL_FIT,
            eval_protocol="test_last_cycle",
        )
        anomaly_test_metrics["pct_iforest_flagged"] = float(test_anomaly_flags.mean())
        if "degradation_roc_auc" in anomaly_test_metrics:
            print(
                f"[{dataset_id}] Test anomaly degradation AUC="
                f"{anomaly_test_metrics['degradation_roc_auc']:.3f} "
                f"(mean score={anomaly_test_metrics['mean_anomaly_score']:.1f})",
                flush=True,
            )

        X_test, y_test = prepare_xy(test_df, feature_cols)
        test_preds = _predict_rul_last_cycle(test_df, best_rul, feature_cols)
        test_metrics = evaluate_rul(y_test.values, test_preds)
        print(
            f"[{dataset_id}] Test RMSE={test_metrics['rmse']:.2f} "
            f"NASA={test_metrics['rul_score']:.2f}",
            flush=True,
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        cox_test_metrics: dict[str, float] = {}
        if cox_model is not None:
            X_test_surv = test_last[cox_model.feature_cols].fillna(0)
            rul_cox_test = cox_model.predict_remaining_rul(
                X_test_surv, test_last["cycle"].values
            )
            cox_test_metrics = evaluate_cox_rul(
                test_last["rul"].values, rul_cox_test
            )
            print(
                f"[{dataset_id}] Cox test RMSE={cox_test_metrics['rmse']:.2f} "
                f"NASA={cox_test_metrics['rul_score']:.2f} "
                f"(winner {winner} NASA={test_metrics['rul_score']:.2f})",
                flush=True,
            )
            for key, value in cox_test_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"test_cox_{key}", float(value))

        for label_col, m in failure_clf_test_metrics.items():
            for key, value in m.items():
                if key == "eval_protocol" or not isinstance(value, (int, float)):
                    continue
                mlflow.log_metric(f"test_{label_col}_{key}", float(value))

        registered_models = register_phase3_models(
            dataset_id,
            winner=winner,
            feature_cols=feature_cols,
            sample_df=train_full,
            training_batch=batch_id,
            best_rul=best_rul,
            failure_clf=failure_clf,
            failure_clf_72=failure_clf_72,
            anomaly_det=anomaly_det,
            cox_model=cox_model,
        )

        fleet = build_fleet_predictions(
            test_df,
            best_rul,
            failure_clf,
            feature_cols,
            model_name=winner,
            failure_clf_72=failure_clf_72,
            anomaly_detector=anomaly_det,
            survival_model=cox_model,
            dataset_id=dataset_id,
        )
        pred_path = PROCESSED_DIR / f"cmapss_{dataset_id}_predictions.parquet"
        fleet.to_parquet(pred_path, index=False)

        summary = {
            "dataset_id": dataset_id,
            "training_batch": batch_id,
            "winner": winner,
            "skip_lstm": skip_lstm,
            "skip_cox": skip_cox,
            "gbm_max_train_rows": gbm_cap,
            "anomaly_max_train_rows": anomaly_cap,
            "val_metrics": val_results,
            "cox_val_metrics": cox_val_metrics,
            "cox_test_metrics": cox_test_metrics,
            "registered_models": registered_models,
            "test_metrics": test_metrics,
            "failure_clf_val_metrics": failure_clf_val_metrics,
            "failure_clf_test_metrics": failure_clf_test_metrics,
            "anomaly_val_metrics": anomaly_val_m,
            "anomaly_test_metrics": anomaly_test_metrics,
            "predictions_path": str(pred_path),
            "train_units": len(train_units),
            "val_units": len(val_units),
        }
        summary_path = ARTIFACTS_DIR / f"cmapss_{dataset_id}_phase3_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_dict(summary, "phase3_summary.json")
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        if mlflow.active_run():
            summary["mlflow_experiment_id"] = mlflow.active_run().info.experiment_id
        summary["mlflow_run_id"] = run_id
        update_training_registry(summary, mlflow_run_id=run_id)

    return summary
