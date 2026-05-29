"""Tests for anomaly detector."""

import pandas as pd

from src.models.anomaly_detector import AnomalyDetector, evaluate_anomaly_degradation_proxy


def test_anomaly_detector_fit_and_score():
    df = pd.DataFrame(
        {
            "rul": [100, 100, 5, 5],
            "sensor_1": [1.0, 1.1, 9.0, 9.5],
            "sensor_2": [2.0, 2.1, 8.0, 8.5],
        }
    )
    det = AnomalyDetector()
    det.fit(df, ["sensor_1", "sensor_2"], min_rul=30)
    scores, flags = det.predict_scores(df)
    assert len(scores) == 4
    assert scores.max() <= 100.0
    assert scores.min() >= 0.0
    assert flags.max() <= 1


def test_evaluate_anomaly_degradation_proxy():
    df = pd.DataFrame({"rul": [100, 10, 5, 80]})
    scores = [10.0, 90.0, 95.0, 20.0]
    m = evaluate_anomaly_degradation_proxy(df, scores, eval_protocol="test")
    assert "degradation_roc_auc" in m
    assert m["degradation_roc_auc"] > 0.5
