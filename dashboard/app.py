"""Industrial Predictive Maintenance System — Streamlit Dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from dashboard.components.databricks_runs import render_databricks_mlflow_panel
from dashboard.components.phase3_metrics import render_registry_banner
from dashboard.data_loader import (
    load_fleet_predictions,
    load_phase3_summary,
    load_training_registry,
    render_dataset_selector,
)
from dashboard.page_init import init_page  # noqa: E402 — after streamlit import

st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_page()

st.title("Industrial Predictive Maintenance System")
st.markdown("Welcome to the fleet health monitoring dashboard.")

_NAV = (
    (
        "pages/01_fleet_overview.py",
        "Fleet Overview",
        "Health scores across all test engines",
    ),
    (
        "pages/02_asset_detail.py",
        "Asset Detail",
        "Sensor trends, RUL, and maintenance briefings",
    ),
    (
        "pages/03_active_alerts.py",
        "Active Alerts",
        "Maintenance alerts from model predictions",
    ),
    (
        "pages/04_model_metrics.py",
        "Model Metrics",
        "MLflow experiment runs (Phase 3)",
    ),
)
st.markdown(
    """
<style>
section[data-testid="stMain"] a[data-testid="stPageLink-NavLink"] {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    text-decoration: none !important;
}
section[data-testid="stMain"] a[data-testid="stPageLink-NavLink"]:hover {
    text-decoration: underline !important;
    opacity: 0.92;
}
</style>
""",
    unsafe_allow_html=True,
)
row_a, row_b = st.columns(2), st.columns(2)
for col, (page_path, label, blurb) in zip(
    (row_a[0], row_a[1], row_b[0], row_b[1]), _NAV
):
    with col:
        st.page_link(page_path, label=label)
        st.caption(blurb)

st.divider()

dataset_id = render_dataset_selector()
fleet = load_fleet_predictions(dataset_id)
summary = load_phase3_summary(dataset_id)
registry = load_training_registry()

render_databricks_mlflow_panel(
    registry,
    expanded=False,
    title="Databricks MLflow experiment (Colab runs)",
)

with st.expander("All FD subsets (training registry)", expanded=False):
    render_registry_banner(registry)

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
            f"**{dataset_id}** · RUL winner: **{summary['winner'].upper()}** · "
            f"NASA score: **{summary['test_metrics']['rul_score']:.2f}** · "
            f"RMSE: **{summary['test_metrics']['rmse']:.2f}**"
        )
else:
    st.info(
        "Run the pipeline: "
        "`python scripts/build_cmapss_dataset.py --dataset FD001` then "
        "`python scripts/train_cmapss_phase3.py`"
    )
