"""Tests for Phase 3 evaluation and model ranking."""

import pandas as pd

from src.models.cmapss_eval import (
    evaluate_failure_classification,
    evaluate_rul,
    last_cycle_per_unit,
    prepare_xy_validation,
    rank_models,
    rul_validation_frame,
)


def test_last_cycle_per_unit():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "cycle": [1, 5, 2, 8],
            "rul": [10, 6, 20, 14],
        }
    )
    last = last_cycle_per_unit(df)
    assert len(last) == 2
    assert last.loc[last["unit_id"] == 1, "cycle"].iloc[0] == 5


def test_rank_models_prefers_lower_nasa_score():
    results = {
        "rf": {"rul_score": 400, "rmse": 18},
        "gbm": {"rul_score": 350, "rmse": 20},
        "lstm": {"rul_score": 360, "rmse": 15},
    }
    assert rank_models(results) == "gbm"


def test_rank_models_tiebreak_rmse_then_simpler():
    results = {
        "rf": {"rul_score": 300, "rmse": 22},
        "gbm": {"rul_score": 300, "rmse": 25},
    }
    assert rank_models(results) == "rf"


def test_rul_validation_frame_drops_terminal_zero_rul():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "rul": [50, 25, 0],
        }
    )
    val = rul_validation_frame(df)
    assert len(val) == 2
    assert (val["rul"] > 0).all()


def test_rank_models_ignores_degenerate_perfect_val():
    results = {
        "rf": {"rul_score": 0.32, "rmse": 3.0},
        "lstm": {"rul_score": 0.0, "rmse": 0.0},
    }
    assert rank_models(results) == "rf"


def test_prepare_xy_validation_excludes_eol():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "cycle": [1, 10, 1, 10],
            "rul": [40, 0, 30, 0],
            "s1": [0.1, 0.2, 0.3, 0.4],
        }
    )
    X, y = prepare_xy_validation(df, ["s1"])
    assert len(X) == 2
    assert list(y) == [40, 30]


def test_evaluate_failure_classification_two_classes():
    y = [0, 0, 1, 1]
    pred = [0, 1, 1, 1]
    proba = [0.1, 0.6, 0.8, 0.9]
    m = evaluate_failure_classification(
        y, pred, proba, eval_protocol="test"
    )
    assert m["n_positive"] == 2
    assert m["n_negative"] == 2
    assert "roc_auc" in m
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_evaluate_failure_classification_single_class_omits_auc():
    m = evaluate_failure_classification(
        [1, 1, 1], [1, 1, 0], [0.9, 0.8, 0.2], eval_protocol="eol_only"
    )
    assert m["n_positive"] == 3
    assert "roc_auc" not in m


def test_evaluate_rul_symmetric_rmse():
    y = [50.0, 50.0]
    pred = [40.0, 60.0]
    m = evaluate_rul(y, pred)
    assert m["rmse"] == 10.0
    assert m["rul_score"] > 0
