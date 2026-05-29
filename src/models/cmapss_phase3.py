"""Phase 3: train, compare, evaluate, and export CMAPSS predictions."""

from __future__ import annotations

import json
import os
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
from src.models.rul_regressor import RULRegressor

load_dotenv()

MODELS_DIR = Path("models")
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed"))
ARTIFACTS_DIR = Path("artifacts")
LSTM_WINDOW = 30
VAL_FRACTION = 0.2
# GBM on full FD002/FD004 row counts can take hours; subsample for fit only.
GBM_MAX_TRAIN_ROWS = 250_000
# UC5 failure windows (cycles as proxy for 24h / 72h planning horizons)
FAILURE_HORIZONS = (30, 72)
ANOMALY_MIN_RUL_FIT = 30
ANOMALY_MAX_TRAIN_ROWS = 100_000


def _subsample_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def _mlflow_setup(dataset_id: str) -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(uri)
    name = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance")
    mlflow.set_experiment(name)
    return name


def _train_rul(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[RULRegressor, dict[str, float], dict[str, float]]:
    fit_df = (
        _subsample_rows(train_df, GBM_MAX_TRAIN_ROWS)
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
) -> None:
    """Fit GBM with balanced sample weights for imbalanced failure horizons."""
    from sklearn.utils.class_weight import compute_sample_weight

    fit_df = _subsample_rows(train_df, GBM_MAX_TRAIN_ROWS)
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
) -> tuple[FailureClassifier, dict[str, float | int | str]]:
    """
    Train failure-within-horizon classifier (UC5 Component B).

    Validation uses non-terminal cycles on held-out engines (mixed 0/1 labels).
    """
    clf = FailureClassifier(model_type="gbm")
    _fit_failure_classifier(clf, train_df, feature_cols, label_col)

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
        _subsample_rows(train_full, GBM_MAX_TRAIN_ROWS)
        if winner == "gbm"
        else train_full
    )
    reg.model.fit(fit_df[feature_cols].fillna(0), fit_df["rul"])
    return reg


def _train_anomaly_detector(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[AnomalyDetector, dict[str, float | int | str]]:
    detector = AnomalyDetector()
    fit_info = detector.fit(
        train_df,
        feature_cols,
        min_rul=ANOMALY_MIN_RUL_FIT,
        max_rows=ANOMALY_MAX_TRAIN_ROWS,
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
) -> dict[str, Any]:
    """
    Full Phase 3 pipeline: engine split, model comparison, test eval, fleet export.
    """
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
        mlflow.log_param("dataset_id", dataset_id)
        mlflow.log_param("val_fraction", val_fraction)
        mlflow.log_param("lstm_window", LSTM_WINDOW)
        mlflow.log_param("train_units", len(train_units))
        mlflow.log_param("val_units", len(val_units))

        for model_type in ("rf", "gbm"):
            print(f"[{dataset_id}] Training RUL {model_type}...", flush=True)
            with mlflow.start_run(run_name=f"{dataset_id}_rul_{model_type}", nested=True):
                reg, val_m, _ = _train_rul(model_type, train_df, val_df, feature_cols)
                mlflow.log_param("model_type", model_type)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
                val_results[model_type] = val_m

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

        winner = rank_models(val_results)
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
                    train_df, val_df, feature_cols, label_col=label_col
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
            winner, train_full, feature_cols, lstm_epochs=lstm_epochs
        )
        failure_clf = FailureClassifier(model_type="gbm")
        _fit_failure_classifier(
            failure_clf, train_full, feature_cols, "failure_30"
        )
        failure_clf_72 = FailureClassifier(model_type="gbm")
        _fit_failure_classifier(
            failure_clf_72, train_full, feature_cols, "failure_72"
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
                train_df, val_df, feature_cols
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
            max_rows=ANOMALY_MAX_TRAIN_ROWS,
        )
        anomaly_det.save(MODELS_DIR / f"anomaly_{dataset_id}.pkl")

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
        for label_col, m in failure_clf_test_metrics.items():
            for key, value in m.items():
                if key == "eval_protocol" or not isinstance(value, (int, float)):
                    continue
                mlflow.log_metric(f"test_{label_col}_{key}", float(value))

        fleet = build_fleet_predictions(
            test_df,
            best_rul,
            failure_clf,
            feature_cols,
            model_name=winner,
            failure_clf_72=failure_clf_72,
            anomaly_detector=anomaly_det,
            dataset_id=dataset_id,
        )
        pred_path = PROCESSED_DIR / f"cmapss_{dataset_id}_predictions.parquet"
        fleet.to_parquet(pred_path, index=False)

        summary = {
            "dataset_id": dataset_id,
            "winner": winner,
            "val_metrics": val_results,
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

    return summary
