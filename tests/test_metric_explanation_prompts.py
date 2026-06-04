"""Tests for Phase 3 metric explanation prompts."""

import json
from pathlib import Path

import pytest

from src.briefings.metric_explanation_prompts import (
    SECTION_COX,
    SECTION_FAILURE,
    SECTION_HEADLINE,
    SECTION_RUL,
    SECTION_SUMMARIZE,
    build_explanation_prompt,
    build_instant_explanation,
    build_section_payload,
    build_summarizer_payload,
    build_summarizer_prompt,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "cmapss_colab_outputs"
    / "artifacts"
    / "cmapss_FD002_phase3_summary.json"
)

_MINIMAL_FD002 = {
    "dataset_id": "FD002",
    "winner": "gbm",
    "test_metrics": {"rmse": 26.4, "rul_score": 13.82, "mae": 22.0},
    "val_metrics": {
        "gbm": {"rmse": 25.0, "rul_score": 19.6, "mae": 20.3},
        "cox": {"rmse": 185.16, "rul_score": 3_387_120_497.0, "mae": 180.0},
    },
    "cox_val_metrics": {
        "concordance": 0.5698,
        "rmse": 185.16,
        "rul_score": 3_387_120_497.0,
    },
    "failure_clf_test_metrics": {
        "failure_30": {"f1": 0.759, "roc_auc": 0.953, "precision": 0.68, "recall": 0.85, "accuracy": 0.87},
    },
    "anomaly_val_metrics": {"mean_anomaly_score": 9.7, "pct_flagged": 0.031, "degradation_roc_auc": 0.573},
}


@pytest.fixture
def fd002_summary() -> dict:
    if _FIXTURE.exists():
        with _FIXTURE.open(encoding="utf-8") as f:
            return json.load(f)
    return _MINIMAL_FD002


def test_build_section_payload_includes_dataset_and_rules(fd002_summary):
    payload = build_section_payload(SECTION_RUL, fd002_summary, "FD002")
    assert payload["dataset_id"] == "FD002"
    assert payload["task"] == SECTION_RUL
    assert "project_rules" in payload
    assert len(payload["rul_comparison"]) == 4
    assert any(row.get("is_winner") for row in payload["rul_comparison"])


def test_cox_payload_compacts_huge_nasa(fd002_summary):
    payload = build_section_payload(SECTION_COX, fd002_summary, "FD002")
    val = payload["cox"]["validation"]
    assert val["nasa"] == "very_large (poor point-RUL fit)"
    assert val["concordance"] == pytest.approx(0.5698, rel=1e-3)


def test_cox_score_readings_fd001_includes_concordance_verdict():
    summary = {
        "winner": "gbm",
        "test_metrics": {"rmse": 23.33, "rul_score": 10.31, "mae": 18.26},
        "cox_val_metrics": {
            "concordance": 0.8483,
            "rmse": float("inf"),
            "rul_score": float("inf"),
        },
        "cox_test_metrics": {"rmse": float("inf"), "rul_score": float("inf")},
    }
    payload = build_section_payload(SECTION_COX, summary, "FD001")
    readings = payload["score_readings"]
    assert readings["validation_concordance"] == pytest.approx(0.8483)
    assert readings["concordance_verdict"] == "strong (≥0.80)"
    assert readings["production_test_rmse"] == pytest.approx(23.33)
    assert "median RUL" in readings["meaning"] or "undefined" in readings["meaning"]
    assert payload["cox"]["validation"]["rmse_note"]


def test_explanation_prompt_includes_score_readings(fd002_summary):
    payload = build_section_payload(SECTION_COX, fd002_summary, "FD002")
    prompt = build_explanation_prompt(SECTION_COX, payload)
    assert "score_readings" in prompt
    assert "interpretation_hints" in prompt
    payload = build_section_payload(SECTION_FAILURE, fd002_summary, "FD002")
    prompt = build_explanation_prompt(SECTION_FAILURE, payload)
    assert "failure classifier" in prompt.lower()
    assert "FD002" in prompt
    assert "failure_30" in prompt or "≤30" in prompt


def test_summarizer_payload_current_fd_only(fd002_summary):
    payload = build_summarizer_payload(fd002_summary, "FD002")
    assert payload["task"] == SECTION_SUMMARIZE
    assert payload["dataset_id"] == "FD002"
    assert "headline" in payload
    assert "rul_comparison" in payload
    assert "FD001" not in json.dumps(payload)


def test_summarizer_prompt(fd002_summary):
    payload = build_summarizer_payload(fd002_summary, "FD002")
    prompt = build_summarizer_prompt(payload)
    assert "executive summary" in prompt.lower()


def test_instant_headline_mentions_winner(fd002_summary):
    text = build_instant_explanation(SECTION_HEADLINE, fd002_summary, "FD002")
    assert "GBM" in text
    assert "FD002" in text


def test_instant_cox_fd001_strong_concordance():
    summary = {
        "winner": "gbm",
        "test_metrics": {"rmse": 23.33, "rul_score": 10.31},
        "cox_val_metrics": {"concordance": 0.8483, "rmse": float("inf"), "rul_score": float("inf")},
        "cox_test_metrics": {"rmse": float("inf")},
    }
    text = build_instant_explanation(SECTION_COX, summary, "FD001")
    assert "0.848" in text
    assert "strong" in text.lower()
    assert "—" in text or "median RUL" in text
    assert "23.33" in text


def test_instant_cox_warns_on_weak_concordance(fd002_summary):
    text = build_instant_explanation(SECTION_COX, fd002_summary, "FD002")
    assert "concordance" in text.lower()
    assert "0.569" in text or "weak" in text.lower()


def test_instant_failure_mentions_horizons(fd002_summary):
    text = build_instant_explanation(SECTION_FAILURE, fd002_summary, "FD002")
    assert "30" in text
    assert "72" in text


def test_instant_summarize_combines_sections(fd002_summary):
    text = build_instant_explanation(SECTION_SUMMARIZE, fd002_summary, "FD002")
    assert "FD002" in text
    assert "GBM" in text
    assert "Anomaly" in text or "anomaly" in text.lower()
