"""Tests for Cox PH survival helpers."""

import numpy as np
import pandas as pd
import pytest

from src.models.cmapss_survival import add_survival_columns, select_cox_features


def test_add_survival_columns_train_marks_terminal_event():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "cycle": [1, 5, 2, 8],
            "rul": [4, 0, 6, 0],
        }
    )
    out = add_survival_columns(df, is_train=True)
    assert out.loc[out["unit_id"] == 1, "event"].tolist() == [0, 1]
    assert out.loc[out["unit_id"] == 2, "event"].tolist() == [0, 1]


def test_add_survival_columns_test_censored():
    df = pd.DataFrame({"unit_id": [1, 1], "cycle": [3, 10], "rul": [20, 13]})
    out = add_survival_columns(df, is_train=False)
    assert out["event"].sum() == 0


def test_select_cox_features_caps_count():
    cols = [f"f{i}" for i in range(50)]
    df = pd.DataFrame({c: np.random.randn(20) for c in cols})
    picked = select_cox_features(df, cols, max_features=10)
    assert len(picked) == 10


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("lifelines"),
    reason="lifelines not installed",
)
def test_survival_model_fit_and_predict_remaining():
    from src.models.survival_model import SurvivalModel

    rng = np.random.default_rng(0)
    rows = []
    for unit in range(8):
        max_c = 10 + unit
        for c in range(1, max_c + 1):
            rows.append(
                {
                    "unit_id": unit,
                    "cycle": c,
                    "duration": float(c),
                    "event": 1 if c == max_c else 0,
                    "f1": rng.normal(),
                    "f2": rng.normal(),
                }
            )
    df = pd.DataFrame(rows)
    model = SurvivalModel(penalizer=0.1)
    model.fit(df, ["f1", "f2"], duration_col="duration", event_col="event")
    last = df.groupby("unit_id").tail(1)
    rem = model.predict_remaining_rul(last, last["cycle"].values)
    assert len(rem) == 8
    assert (rem >= 0).all()
    prob = model.predict_survival_probability(last, last["cycle"].values, horizon=5)
    assert (prob >= 0).all() and (prob <= 1).all()
