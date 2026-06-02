"""Log and register CMAPSS Phase 3 models in MLflow (Databricks Model Registry)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

MODELS_DIR = Path("models")


def registry_enabled() -> bool:
    return os.getenv("MLFLOW_REGISTER_MODELS", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def use_legacy_model_registry() -> bool:
    """Workspace Model Registry (short names). Set MLFLOW_USE_LEGACY_MODEL_REGISTRY=1."""
    return os.getenv("MLFLOW_USE_LEGACY_MODEL_REGISTRY", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _short_model_name(role: str, dataset_id: str, *, variant: str | None = None) -> str:
    parts = ["cmapss", role]
    if variant:
        parts.append(variant)
    parts.append(dataset_id)
    return "_".join(parts)


def registered_model_name(role: str, dataset_id: str, *, variant: str | None = None) -> str:
    """
    Registered model name for MLflow.

    - Unity Catalog (default on new Databricks): ``catalog.schema.cmapss_rul_gbm_FD001``
    - Legacy workspace registry: ``cmapss_rul_gbm_FD001`` (set MLFLOW_USE_LEGACY_MODEL_REGISTRY=1)
    """
    short = _short_model_name(role, dataset_id, variant=variant)
    if use_legacy_model_registry():
        return short
    catalog = os.getenv("MLFLOW_UC_CATALOG", "main").strip()
    schema = os.getenv("MLFLOW_UC_SCHEMA", "default").strip()
    return f"{catalog}.{schema}.{short}"


def configure_model_registry() -> str:
    """Set registry URI before log_model(registered_model_name=...). Returns active URI."""
    import mlflow

    if use_legacy_model_registry():
        uri = "databricks"
    else:
        uri = os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc").strip() or "databricks-uc"
    mlflow.set_registry_uri(uri)
    return uri


def _log_sklearn(
    sk_model: Any,
    name: str,
    *,
    feature_cols: list[str],
    sample_df: pd.DataFrame,
    extra_metadata: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    import mlflow
    from mlflow.models import infer_signature

    try:
        import mlflow.sklearn
    except ImportError:
        return None

    X = sample_df[feature_cols].fillna(0)
    signature = infer_signature(X, sk_model.predict(X))
    metadata = {"feature_cols": json.dumps(feature_cols)}
    if extra_metadata:
        metadata.update(extra_metadata)

    info = mlflow.sklearn.log_model(
        sk_model=sk_model,
        name=name,
        registered_model_name=name,
        signature=signature,
        input_example=X.head(3),
        metadata=metadata,
    )
    version = getattr(info, "registered_model_version", None)
    return {
        "name": name,
        "version": str(version) if version is not None else "",
        "model_uri": getattr(info, "model_uri", ""),
    }


def _log_survival_pyfunc(
    path: Path,
    name: str,
    *,
    feature_cols: list[str],
    extra_metadata: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    import mlflow

    if not path.exists():
        return None

    metadata = {"feature_cols": json.dumps(feature_cols)}
    if extra_metadata:
        metadata.update(extra_metadata)

    class SurvivalPyFunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context) -> None:
            from src.models.survival_model import SurvivalModel

            self.model = SurvivalModel.load(context.artifacts["bundle"])

        def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
            if "cycle" not in model_input.columns:
                raise ValueError("column 'cycle' is required for Cox survival scoring")
            cols = self.model.feature_cols
            X = model_input[cols].fillna(0)
            preds = self.model.predict_remaining_rul(X, model_input["cycle"].values)
            return pd.DataFrame({"rul_pred_cox": preds})

    try:
        info = mlflow.pyfunc.log_model(
            python_model=SurvivalPyFunc(),
            artifacts={"bundle": str(path.resolve())},
            name=name,
            registered_model_name=name,
            metadata=metadata,
        )
    except Exception as exc:
        print(f"  Registry survival skip {name}: {exc}", flush=True)
        return None

    version = getattr(info, "registered_model_version", None)
    return {
        "name": name,
        "version": str(version) if version is not None else "",
        "model_uri": getattr(info, "model_uri", ""),
    }


def register_phase3_models(
    dataset_id: str,
    *,
    winner: str,
    feature_cols: list[str],
    sample_df: pd.DataFrame,
    training_batch: str,
    best_rul: Any,
    failure_clf: Any,
    failure_clf_72: Any,
    anomaly_det: Any,
    cox_model: Any | None = None,
) -> dict[str, dict[str, str]]:
    """
    Register production pickles into MLflow Model Registry under the active parent run.

    Skips quietly when MLFLOW_REGISTER_MODELS=0 or registration fails (permissions).
    """
    if not registry_enabled():
        return {}

    registry_uri = configure_model_registry()
    print(
        f"[{dataset_id}] Model registry URI={registry_uri!r} "
        f"(legacy names={use_legacy_model_registry()})",
        flush=True,
    )

    registered: dict[str, dict[str, str]] = {}
    meta = {
        "dataset_id": dataset_id,
        "training_batch": training_batch,
        "pipeline": "cmapss_phase3",
    }

    if winner == "lstm" and hasattr(best_rul, "net"):
        import mlflow.pytorch

        name = registered_model_name("rul", dataset_id, variant="lstm")
        try:
            info = mlflow.pytorch.log_model(
                pytorch_model=best_rul.net,
                name=name,
                registered_model_name=name,
                metadata={
                    **meta,
                    "sequence_length": str(getattr(best_rul, "sequence_length", 30)),
                    "input_size": str(getattr(best_rul, "input_size", len(feature_cols))),
                    "feature_cols": json.dumps(feature_cols),
                },
            )
            registered["rul"] = {
                "name": name,
                "version": str(getattr(info, "registered_model_version", "")),
                "model_uri": getattr(info, "model_uri", ""),
            }
        except Exception as exc:
            print(f"  Registry skip {name}: {exc}", flush=True)

    elif hasattr(best_rul, "model"):
        name = registered_model_name("rul", dataset_id, variant=winner)
        entry = _log_sklearn(
            best_rul.model,
            name,
            feature_cols=feature_cols,
            sample_df=sample_df,
            extra_metadata={**meta, "winner": winner, "role": "rul"},
        )
        if entry:
            registered["rul"] = entry

    for role, clf in (("failure_30", failure_clf), ("failure_72", failure_clf_72)):
        name = registered_model_name(role, dataset_id)
        entry = _log_sklearn(
            clf.model,
            name,
            feature_cols=feature_cols,
            sample_df=sample_df,
            extra_metadata={**meta, "role": role},
        )
        if entry:
            registered[role] = entry

    name = registered_model_name("anomaly", dataset_id)
    entry = _log_sklearn(
        anomaly_det.model,
        name,
        feature_cols=feature_cols,
        sample_df=sample_df,
        extra_metadata={**meta, "role": "anomaly", "min_rul_fit": str(anomaly_det.min_rul_fit)},
    )
    if entry:
        registered["anomaly"] = entry

    if cox_model is not None and cox_model.model is not None:
        surv_path = MODELS_DIR / f"survival_{dataset_id}.pkl"
        name = registered_model_name("survival", dataset_id)
        entry = _log_survival_pyfunc(
            surv_path,
            name,
            feature_cols=cox_model.feature_cols,
            extra_metadata={**meta, "role": "cox_ph"},
        )
        if entry:
            registered["survival"] = entry

    if registered:
        print(
            f"[{dataset_id}] Registered {len(registered)} model(s) in MLflow: "
            + ", ".join(f"{v['name']}@v{v['version']}" for v in registered.values()),
            flush=True,
        )
    else:
        print(f"[{dataset_id}] No models registered (see errors above).", flush=True)

    return registered


def register_models_from_disk(
    dataset_id: str,
    *,
    winner: str | None = None,
    feature_cols: list[str] | None = None,
    sample_df: pd.DataFrame | None = None,
    training_batch: str | None = None,
) -> dict[str, dict[str, str]]:
    """
    Register models already saved under ``models/`` (e.g. after Colab train, new cell).

    Starts a short MLflow run if none is active.
    """
    import mlflow

    from src.models.anomaly_detector import AnomalyDetector
    from src.models.cmapss_eval import load_feature_columns
    from src.models.failure_classifier import FailureClassifier
    from src.models.lstm_model import LSTMModel
    from src.models.rul_regressor import RULRegressor
    from src.models.survival_model import SurvivalModel

    artifacts = Path("artifacts")
    if winner is None:
        summary_path = artifacts / f"cmapss_{dataset_id}_phase3_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            winner = summary.get("winner", "gbm")
            training_batch = training_batch or summary.get("training_batch")
        else:
            winner = "gbm"

    if feature_cols is None:
        feature_cols = load_feature_columns(artifacts, dataset_id)
    if sample_df is None:
        train_path = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")) / f"cmapss_{dataset_id}_train.parquet"
        sample_df = pd.read_parquet(train_path).head(200)
    training_batch = training_batch or os.getenv("MLFLOW_TRAINING_BATCH_ID", "register_from_disk")

    lstm_path = MODELS_DIR / f"rul_lstm_{dataset_id}.pt"
    if winner == "lstm" and lstm_path.exists():
        best_rul = LSTMModel.load(lstm_path)
    else:
        pkl = MODELS_DIR / f"rul_{winner}_{dataset_id}.pkl"
        if not pkl.exists():
            for fallback in ("gbm", "rf"):
                pkl = MODELS_DIR / f"rul_{fallback}_{dataset_id}.pkl"
                if pkl.exists():
                    winner = fallback
                    break
        best_rul = RULRegressor.load(pkl)

    failure_clf = FailureClassifier.load(MODELS_DIR / f"failure_30_{dataset_id}.pkl")
    failure_clf_72 = FailureClassifier.load(MODELS_DIR / f"failure_72_{dataset_id}.pkl")
    anomaly_det = AnomalyDetector.load(MODELS_DIR / f"anomaly_{dataset_id}.pkl")
    surv_path = MODELS_DIR / f"survival_{dataset_id}.pkl"
    cox_model = SurvivalModel.load(surv_path) if surv_path.exists() else None

    active = mlflow.active_run()
    if active is None:
        with mlflow.start_run(run_name=f"{dataset_id}_registry_only"):
            mlflow.set_tags(
                {
                    "pipeline": "cmapss_phase3",
                    "dataset_id": dataset_id,
                    "training_batch": training_batch,
                    "task": "model_registry",
                }
            )
            return register_phase3_models(
                dataset_id,
                winner=winner,
                feature_cols=feature_cols,
                sample_df=sample_df,
                training_batch=training_batch,
                best_rul=best_rul,
                failure_clf=failure_clf,
                failure_clf_72=failure_clf_72,
                anomaly_det=anomaly_det,
                cox_model=cox_model,
            )

    return register_phase3_models(
        dataset_id,
        winner=winner,
        feature_cols=feature_cols,
        sample_df=sample_df,
        training_batch=training_batch,
        best_rul=best_rul,
        failure_clf=failure_clf,
        failure_clf_72=failure_clf_72,
        anomaly_det=anomaly_det,
        cox_model=cox_model,
    )
