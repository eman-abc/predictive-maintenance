"""Feature column resolution when artifacts JSON is missing."""

import json

import pandas as pd

from src.models.cmapss_eval import infer_feature_columns_from_frame, load_feature_columns


def test_infer_feature_columns_from_frame():
    df = pd.DataFrame(
        {
            "unit_id": [1],
            "cycle": [1],
            "rul": [10.0],
            "sensor_1": [0.1],
            "rolling_mean_sensor_1": [0.2],
        }
    )
    cols = infer_feature_columns_from_frame(df)
    assert "rolling_mean_sensor_1" in cols
    assert "sensor_1" not in cols


def test_load_feature_columns_from_pickle(tmp_path, monkeypatch):
    import joblib

    monkeypatch.setattr("src.models.cmapss_eval.MODELS_DIR", tmp_path, raising=False)
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(
        {"model": None, "feature_cols": ["f1", "f2"], "type": "gbm"},
        models / "rul_gbm_FD001.pkl",
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    cols = load_feature_columns(artifacts, "FD001", models_dir=models)
    assert cols == ["f1", "f2"]
