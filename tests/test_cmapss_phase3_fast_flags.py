"""Tests for Phase 3 fast-training CLI parameters."""

from src.models import cmapss_phase3 as p3


def test_run_phase3_accepts_fast_kwargs():
    """Signature includes skip_lstm and row caps (smoke — no training)."""
    import inspect

    sig = inspect.signature(p3.run_phase3)
    assert "skip_lstm" in sig.parameters
    assert "gbm_max_rows" in sig.parameters
    assert "anomaly_max_rows" in sig.parameters


def test_run_phase3_all_forwards_kwargs():
    import inspect

    sig = inspect.signature(p3.run_phase3_all)
    assert sig.parameters["skip_lstm"].default is False
    assert sig.parameters["gbm_max_rows"].default is None
