"""Sensor time-series chart widget for Streamlit."""

import streamlit as st
import pandas as pd
from src.utils.visualizations import plot_sensor_timeseries


def render_sensor_chart(
    df: pd.DataFrame, asset_id: str, sensors: list[str] | None = None
) -> None:
    """Render sensor time-series chart."""
    unit_id = int(asset_id.split("-")[1]) if "-" in asset_id else asset_id
    if "unit_id" not in df.columns:
        df = df.copy()
        df["unit_id"] = unit_id
    fig = plot_sensor_timeseries(df, unit_id if isinstance(unit_id, int) else 1, sensors=sensors)
    st.plotly_chart(fig, use_container_width=True)
