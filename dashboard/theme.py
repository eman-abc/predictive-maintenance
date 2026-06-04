"""Streamlit global theme — palette, semantic UI, and chart helpers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.chart_theme import (
    COLOR_BG,
    COLOR_CAUTION,
    COLOR_CRITICAL,
    COLOR_HEALTHY,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    PALACE_PURPLE,
    PRISMARINE,
    RIP_VAN_PERIWINKLE,
    TELOPEA,
    alert_level_color,
    apply_plotly_theme,
    health_bar_color,
)

STREAMLIT_CSS = f"""
<style>
    /* Base */
    .stApp {{
        background-color: {COLOR_BG};
    }}
    [data-testid="stSidebar"] {{
        background-color: {TELOPEA};
        border-right: 1px solid {PALACE_PURPLE};
    }}
    h1, h2, h3, h4, [data-testid="stMarkdownContainer"] p {{
        color: {COLOR_TEXT};
    }}
    [data-testid="stCaptionContainer"] {{
        color: {COLOR_TEXT_MUTED};
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background: {TELOPEA};
        border: 1px solid {PALACE_PURPLE};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLOR_TEXT_MUTED};
    }}
    [data-testid="stMetricValue"] {{
        color: {COLOR_TEXT};
    }}

    /* Semantic alerts — Streamlit native boxes */
    [data-testid="stNotification"][data-baseweb="notification"] {{
        border-radius: 8px;
    }}

    /* Dataframes */
    [data-testid="stDataFrame"] {{
        border: 1px solid {PALACE_PURPLE};
        border-radius: 8px;
    }}

    /* Primary button */
    .stButton > button[kind="primary"] {{
        background-color: {PRISMARINE};
        border-color: {PRISMARINE};
        color: {COLOR_TEXT};
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #14828B;
        border-color: #14828B;
    }}

    /* Legend strip for operators */
    .pm-legend {{
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        margin: 0 0 1rem 0;
        padding: 10px 14px;
        background: {TELOPEA};
        border-radius: 8px;
        border: 1px solid {PALACE_PURPLE};
        font-size: 0.85rem;
    }}
    .pm-legend span {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: {COLOR_TEXT_MUTED};
    }}
    .pm-swatch {{
        width: 12px;
        height: 12px;
        border-radius: 3px;
        display: inline-block;
    }}
</style>
"""


def apply_streamlit_theme() -> None:
    """Inject global CSS once per session."""
    if st.session_state.get("_pm_theme_applied"):
        return
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    st.session_state["_pm_theme_applied"] = True


def render_semantic_legend() -> None:
    """Operator color key — healthy / caution / critical."""
    st.markdown(
        f"""
        <div class="pm-legend">
          <span><i class="pm-swatch" style="background:{COLOR_HEALTHY}"></i> Healthy / nominal</span>
          <span><i class="pm-swatch" style="background:{COLOR_CAUTION}"></i> Caution / trending</span>
          <span><i class="pm-swatch" style="background:{COLOR_CRITICAL}"></i> Critical / act now</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fleet_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Semantic row colors from alert_level or health_score."""

    def _row_style(row: pd.Series):
        level = str(row.get("status", row.get("alert_level", "normal"))).lower()
        if level == "critical":
            bg = f"rgba(157, 44, 11, 0.35)"
        elif level == "warning":
            bg = f"rgba(145, 164, 215, 0.22)"
        else:
            hs = row.get("health_score", 100)
            try:
                hs = float(hs)
            except (TypeError, ValueError):
                hs = 100
            if hs < 40:
                bg = "rgba(157, 44, 11, 0.25)"
            elif hs < 70:
                bg = "rgba(145, 164, 215, 0.18)"
            else:
                bg = "rgba(15, 114, 122, 0.2)"
        return [f"background-color: {bg}; color: {COLOR_TEXT}"] * len(row)

    subset = [c for c in df.columns if c in ("health_score", "status", "alert_level")]
    styler = df.style.apply(_row_style, axis=1)
    if "health_score" in df.columns:
        styler = styler.bar(subset=["health_score"], color=COLOR_HEALTHY, vmin=0, vmax=100)
    return styler


def fleet_health_bar_chart(df: pd.DataFrame, *, health_col: str = "health_score") -> go.Figure:
    """Bar chart with per-bar semantic color."""
    if health_col not in df.columns:
        health_col = "health_score"
    assets = df["asset_id"].astype(str).tolist()
    scores = df[health_col].astype(float).tolist()
    colors = [health_bar_color(s) for s in scores]
    fig = go.Figure(
        go.Bar(
            x=assets,
            y=scores,
            marker_color=colors,
            marker_line={"color": COLOR_TEXT_MUTED, "width": 0.5},
        )
    )
    fig.update_layout(
        title="Fleet health score",
        xaxis_title="Asset",
        yaxis_title="Health score",
        yaxis={"range": [0, 100]},
    )
    return apply_plotly_theme(fig)


def metric_delta_color(level: str) -> str | None:
    """Hint for custom metric styling."""
    return alert_level_color(level)
