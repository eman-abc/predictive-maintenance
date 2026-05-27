"""Health score gauge widget for Streamlit."""

import streamlit as st
from src.utils.visualizations import plot_health_gauge


def render_health_gauge(score: float, title: str = "Health Score") -> None:
    """Render an interactive health gauge in Streamlit."""
    fig = plot_health_gauge(score, title=title)
    st.plotly_chart(fig, use_container_width=True)
