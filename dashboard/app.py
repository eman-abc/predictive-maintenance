"""Industrial Predictive Maintenance System — Streamlit Dashboard."""

import streamlit as st

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

- **Fleet Overview** — health scores across all assets
- **Asset Detail** — per-asset sensor trends and RUL
- **Active Alerts** — manage and acknowledge alerts
- **Model Metrics** — compare MLflow experiment runs
"""
)

col1, col2, col3 = st.columns(3)
col1.metric("Fleet Health", "82%", "+3%")
col2.metric("Active Alerts", "4", "-2")
col3.metric("Assets Monitored", "100", "0")

st.info("Download datasets to `data/raw/` and run `python -m src.models.train` to populate live metrics.")
