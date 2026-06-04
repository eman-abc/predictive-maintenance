"""Phase 3 summary tables for Model Metrics and home page."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "artifacts" / "cmapss_training_registry.json"


def _fmt_metric(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(v) or math.isnan(v):
        return "—"
    return f"{v:.{digits}f}"


def load_training_registry() -> dict | None:
    if not REGISTRY_PATH.exists():
        return None
    with REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def registry_overview_table(registry: dict) -> pd.DataFrame:
    rows = []
    for ds_id, entry in registry.get("datasets", {}).items():
        rows.append(
            {
                "dataset": ds_id,
                "winner": entry.get("winner", "").upper(),
                "test_rmse": _fmt_metric(entry.get("test_rmse")),
                "test_nasa": _fmt_metric(entry.get("test_nasa_score")),
                "cox_rmse": _fmt_metric(entry.get("test_cox_rmse")),
                "cox_nasa": _fmt_metric(entry.get("test_cox_nasa_score")),
                "trained_at": (entry.get("trained_at") or "")[:10],
            }
        )
    return pd.DataFrame(rows)


def rul_model_comparison(summary: dict) -> pd.DataFrame:
    val = summary.get("val_metrics") or {}
    rows = []
    for model in ("rf", "gbm", "lstm", "cox"):
        m = val.get(model) or {}
        if not m:
            continue
        rows.append(
            {
                "model": model.upper(),
                "val_rmse": _fmt_metric(m.get("rmse")),
                "val_nasa": _fmt_metric(m.get("rul_score")),
                "val_mae": _fmt_metric(m.get("mae")),
            }
        )
    winner = (summary.get("winner") or "").upper()
    df = pd.DataFrame(rows)
    if not df.empty and winner:
        df["winner"] = df["model"].eq(winner).map({True: "✓", False: ""})
    return df


def cox_detail(summary: dict) -> pd.DataFrame:
    rows = []
    for label, block in (
        ("validation", summary.get("cox_val_metrics")),
        ("test", summary.get("cox_test_metrics")),
    ):
        if not block:
            continue
        rows.append(
            {
                "split": label,
                "concordance": _fmt_metric(block.get("concordance"), digits=3),
                "rmse": _fmt_metric(block.get("rmse")),
                "nasa": _fmt_metric(block.get("rul_score")),
            }
        )
    return pd.DataFrame(rows)


def failure_classifier_table(summary: dict, split: str = "failure_clf_test_metrics") -> pd.DataFrame:
    block = summary.get(split) or {}
    rows = []
    for horizon in ("failure_30", "failure_72"):
        m = block.get(horizon) or {}
        if not m:
            continue
        rows.append(
            {
                "horizon": horizon.replace("failure_", "≤") + " cycles",
                "f1": _fmt_metric(m.get("f1"), digits=3),
                "roc_auc": _fmt_metric(m.get("roc_auc"), digits=3),
                "precision": _fmt_metric(m.get("precision"), digits=3),
                "recall": _fmt_metric(m.get("recall"), digits=3),
                "accuracy": _fmt_metric(m.get("accuracy"), digits=3),
            }
        )
    return pd.DataFrame(rows)


def anomaly_metrics_table(summary: dict) -> pd.DataFrame:
    rows = []
    for split, key in (("validation", "anomaly_val_metrics"), ("test", "anomaly_test_metrics")):
        m = summary.get(key) or {}
        if not m:
            continue
        rows.append(
            {
                "split": split,
                "mean_score": _fmt_metric(m.get("mean_anomaly_score"), digits=1),
                "pct_flagged": _fmt_metric(m.get("pct_flagged"), digits=3),
                "degradation_auc": _fmt_metric(m.get("degradation_roc_auc"), digits=3),
            }
        )
    return pd.DataFrame(rows)


def render_phase3_summary(
    summary: dict,
    dataset_id: str,
    *,
    registry_entry: dict | None = None,
) -> None:
    """Full Phase 3 metrics panel for one FD subset."""
    from dashboard.components.metric_explanation_panel import (
        SECTION_ANOMALY,
        SECTION_COX,
        SECTION_FAILURE,
        SECTION_HEADLINE,
        SECTION_RUL,
        render_metric_explanation_controls,
        render_summarize_all_controls,
    )

    st.subheader(f"{dataset_id} — Phase 3 results")
    st.caption(
        "Click **Explain** on any block for an Ollama interpretation (each click replaces the previous text)."
    )
    test = summary.get("test_metrics") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test RMSE (winner)", _fmt_metric(test.get("rmse")))
    c2.metric("Test NASA score", _fmt_metric(test.get("rul_score")))
    c3.metric("RUL winner", (summary.get("winner") or "—").upper())
    flags = []
    if summary.get("skip_lstm"):
        flags.append("LSTM skipped")
    if summary.get("skip_cox"):
        flags.append("Cox skipped")
    c4.metric("Training", flags[0] if flags else "full pipeline")

    st.markdown("**Headline test results**")
    render_metric_explanation_controls(
        SECTION_HEADLINE,
        dataset_id,
        summary,
        registry_entry=registry_entry,
        button_prefix="headline",
    )

    st.markdown("**RUL model comparison (validation)**")
    st.dataframe(rul_model_comparison(summary), use_container_width=True, hide_index=True)
    render_metric_explanation_controls(
        SECTION_RUL,
        dataset_id,
        summary,
        registry_entry=registry_entry,
    )

    cox_df = cox_detail(summary)
    if not cox_df.empty:
        st.markdown("**Cox proportional hazards (survival)**")
        st.dataframe(cox_df, use_container_width=True, hide_index=True)
        survival_path = ROOT / "models" / f"survival_{dataset_id}.pkl"
        if survival_path.exists():
            st.caption(f"Survival model: `{survival_path.name}`")
        render_metric_explanation_controls(
            SECTION_COX,
            dataset_id,
            summary,
            registry_entry=registry_entry,
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Failure classifiers (test)**")
        st.dataframe(failure_classifier_table(summary), use_container_width=True, hide_index=True)
        render_metric_explanation_controls(
            SECTION_FAILURE,
            dataset_id,
            summary,
            registry_entry=registry_entry,
            button_prefix="failure",
        )
    with col_b:
        st.markdown("**Anomaly detection (Isolation Forest)**")
        st.dataframe(anomaly_metrics_table(summary), use_container_width=True, hide_index=True)
        render_metric_explanation_controls(
            SECTION_ANOMALY,
            dataset_id,
            summary,
            registry_entry=registry_entry,
            button_prefix="anomaly",
        )

    st.divider()
    render_summarize_all_controls(
        dataset_id,
        summary,
        registry_entry=registry_entry,
    )

    with st.expander("Raw summary JSON"):
        st.json(summary)


def render_registry_banner(registry: dict | None) -> None:
    if not registry:
        return
    st.markdown("**All CMAPSS subsets (Colab training registry)**")
    st.dataframe(registry_overview_table(registry), use_container_width=True, hide_index=True)
    batch = registry.get("datasets", {}).get("FD001", {}).get("training_batch")
    if batch:
        st.caption(f"Training batch: `{batch}` · updated {registry.get('updated_at', '')[:10]}")
