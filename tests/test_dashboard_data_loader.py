"""Tests for dashboard dataset discovery and prediction column aliases."""

from dashboard.data_loader import list_datasets_with_predictions, predictions_path


def test_list_datasets_with_predictions_includes_fd001_when_present():
    available = list_datasets_with_predictions()
    if predictions_path("FD001").exists():
        assert "FD001" in available
