"""Industrial Predictive Maintenance System — Streamlit Dashboard."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.data_loader import (
    load_fleet_predictions,
    load_phase3_summary,
    render_dataset_selector,
)

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Industrial Predictive Maintenance System")
st.markdown(
    """
Welcome to the fleet health monitoring dashboard. Use the sidebar to navigate:

- **Fleet Overview** — health scores across all test engines
- **Asset Detail** — per-asset sensor trends, RUL, and Ollama maintenance briefings
- **Active Alerts** — maintenance alerts from model predictions
- **Model Metrics** — MLflow experiment runs (Phase 3)
"""
)

dataset_id = render_dataset_selector()
fleet = load_fleet_predictions(dataset_id)
summary = load_phase3_summary(dataset_id)

if fleet is not None:
    critical = int((fleet["alert_level"] == "critical").sum())
    warning = int((fleet["alert_level"] == "warning").sum())
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Fleet Health (avg)", f"{fleet['health_score'].mean():.1f}%")
    col2.metric("Critical Alerts", critical)
    col3.metric("Warning Alerts", warning)
    col4.metric("Test Engines", len(fleet))
    if "anomaly_score" in fleet.columns:
        col5.metric("Avg Anomaly", f"{fleet['anomaly_score'].mean():.1f}")
    else:
        col5.metric("Avg Anomaly", "—")
    if summary:
        st.success(
            f"**{dataset_id}** — RUL winner: **{summary['winner'].upper()}** | "
            f"NASA score: **{summary['test_metrics']['rul_score']:.2f}** | "
            f"RMSE: **{summary['test_metrics']['rmse']:.2f}**"
        )
else:
    st.info(
        "Run the pipeline: "
        "`python scripts/build_cmapss_dataset.py --dataset FD001` then "
        "`python scripts/train_cmapss_phase3.py`"
    )
