"""RUL trend chart widget for Streamlit."""

import streamlit as st
import pandas as pd
from src.utils.visualizations import plot_rul_trend


def render_rul_chart(df: pd.DataFrame, asset_id: str, rul_col: str = "rul") -> None:
    """Render RUL degradation chart."""
    unit_id = int(asset_id.split("-")[1]) if "-" in asset_id else asset_id
    if "unit_id" not in df.columns:
        df = df.copy()
        df["unit_id"] = unit_id
    fig = plot_rul_trend(df, unit_id if isinstance(unit_id, int) else 1, rul_col=rul_col)
    st.plotly_chart(fig, use_container_width=True)
