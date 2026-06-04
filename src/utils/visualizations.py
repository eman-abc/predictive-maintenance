"""Reusable plotting functions for notebooks and dashboard."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from src.utils.chart_theme import (
    COLOR_CAUTION_LINE,
    COLOR_CRITICAL,
    COLOR_CAUTION,
    COLOR_HEALTHY,
    COLOR_HEALTHY_LINE,
    COLOR_RUL_LINE,
    COLOR_SURVIVAL_LINE,
    COLOR_TEXT_MUTED,
    SENSOR_LINE_COLORS,
    apply_plotly_theme,
    health_gauge_color,
)


def plot_rul_trend(df: pd.DataFrame, unit_id: int, rul_col: str = "rul") -> go.Figure:
    """Plot RUL degradation trend for a single asset."""
    unit_df = df[df["unit_id"] == unit_id]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=unit_df["cycle"],
            y=unit_df[rul_col],
            name="RUL",
            mode="lines",
            line={"color": COLOR_RUL_LINE, "width": 2.5},
        )
    )
    fig.update_layout(
        title=f"RUL Trend — Unit {unit_id}",
        xaxis_title="Operating Cycle",
        yaxis_title="Remaining Useful Life",
    )
    return apply_plotly_theme(fig)


def plot_sensor_timeseries(
    df: pd.DataFrame, unit_id: int, sensors: list[str] | None = None
) -> go.Figure:
    """Plot sensor readings over time for a single asset."""
    unit_df = df[df["unit_id"] == unit_id]
    sensors = sensors or [f"sensor_{i}" for i in range(1, 6)]
    fig = go.Figure()
    for i, sensor in enumerate(sensors):
        if sensor in unit_df.columns:
            color = SENSOR_LINE_COLORS[i % len(SENSOR_LINE_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=unit_df["cycle"],
                    y=unit_df[sensor],
                    name=sensor,
                    mode="lines",
                    line={"color": color, "width": 2},
                )
            )
    fig.update_layout(
        title=f"Sensor Readings — Unit {unit_id}",
        xaxis_title="Cycle",
        yaxis_title="Sensor Value",
    )
    return apply_plotly_theme(fig)


def plot_survival_curve(curve_df: pd.DataFrame, asset_id: str) -> go.Figure:
    """Conditional survival probability from current cycle (Cox PH)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve_df["cycle"],
            y=curve_df["survival_prob"],
            name="Survival",
            mode="lines",
            line={"color": COLOR_SURVIVAL_LINE, "width": 2.5},
        )
    )
    fig.update_layout(
        title=f"Survival curve — {asset_id}",
        xaxis_title="Future operating cycle",
        yaxis_title="P(operational | survived to current cycle)",
        yaxis={"range": [0, 1.05]},
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=COLOR_CAUTION_LINE,
        line_width=2,
        annotation_text="50%",
        annotation_font_color=COLOR_TEXT_MUTED,
    )
    return apply_plotly_theme(fig)


def plot_health_gauge(score: float, title: str = "Health Score") -> go.Figure:
    """Create a gauge chart for asset health score (0-100)."""
    bar_color = health_gauge_color(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"color": "#EDE8F2"}},
            number={"font": {"color": "#EDE8F2"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLOR_TEXT_MUTED},
                "bar": {"color": bar_color, "thickness": 0.85},
                "bgcolor": "#110B0F",
                "bordercolor": "#68477C",
                "steps": [
                    {"range": [0, 40], "color": "rgba(157, 44, 11, 0.45)"},
                    {"range": [40, 70], "color": "rgba(145, 164, 215, 0.35)"},
                    {"range": [70, 100], "color": "rgba(15, 114, 122, 0.45)"},
                ],
                "threshold": {
                    "line": {"color": COLOR_CRITICAL, "width": 2},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=300)
    return apply_plotly_theme(fig)


def plot_confusion_matrix_heatmap(cm, labels: list[str] | None = None) -> plt.Figure:
    """Matplotlib heatmap for classification confusion matrix."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig
