"""Per-asset deep-dive view."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.components.health_gauge import render_health_gauge
from dashboard.components.rul_chart import render_rul_chart
from dashboard.components.sensor_chart import render_sensor_chart

st.set_page_config(page_title="Asset Detail", layout="wide")
st.title("Asset Detail")

asset_id = st.selectbox("Select Asset", [f"ENG-{i:03d}" for i in range(1, 21)])

np.random.seed(int(asset_id.split("-")[1]))
cycles = np.arange(1, 101)
rul = np.maximum(125 - cycles + np.random.normal(0, 3, len(cycles)), 0)
health = max(0, min(100, rul[-1] / 125 * 100))

col1, col2 = st.columns([1, 2])
with col1:
    render_health_gauge(health, title=f"{asset_id} Health")
    st.metric("Predicted RUL", f"{rul[-1]:.0f} cycles")
    st.metric("Failure Probability", f"{np.random.uniform(0.05, 0.4):.0%}")

with col2:
    trend_df = pd.DataFrame({"cycle": cycles, "rul": rul})
    render_rul_chart(trend_df, asset_id)

sensor_df = pd.DataFrame({
    "cycle": cycles,
    "sensor_1": 600 + cycles * 0.5 + np.random.normal(0, 2, len(cycles)),
    "sensor_2": 1400 + cycles * 1.2 + np.random.normal(0, 5, len(cycles)),
    "sensor_3": 1100 - cycles * 0.8 + np.random.normal(0, 3, len(cycles)),
})
render_sensor_chart(sensor_df, asset_id)
