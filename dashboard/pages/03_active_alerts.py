"""Active alerts management view."""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Active Alerts", layout="wide")
st.title("Active Alerts")

alerts = pd.DataFrame([
    {"alert_id": "ALT-000001", "asset_id": "ENG-003", "level": "critical", "rul": 8, "health_score": 32, "status": "open"},
    {"alert_id": "ALT-000002", "asset_id": "ENG-007", "level": "warning", "rul": 22, "health_score": 55, "status": "open"},
    {"alert_id": "ALT-000003", "asset_id": "ENG-012", "level": "warning", "rul": 28, "health_score": 48, "status": "acknowledged"},
    {"alert_id": "ALT-000004", "asset_id": "ENG-019", "level": "critical", "rul": 5, "health_score": 21, "status": "open"},
])

level_filter = st.multiselect("Filter by Level", ["critical", "warning"], default=["critical", "warning"])
filtered = alerts[alerts["level"].isin(level_filter)]

st.dataframe(filtered, use_container_width=True, hide_index=True)

if st.button("Generate Shift Briefing"):
    st.info("Connect Ollama to generate AI-powered shift handover briefings.")
