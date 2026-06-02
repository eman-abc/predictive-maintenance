"""Reusable plotting functions for notebooks and dashboard."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_rul_trend(df: pd.DataFrame, unit_id: int, rul_col: str = "rul") -> go.Figure:
    """Plot RUL degradation trend for a single asset."""
    unit_df = df[df["unit_id"] == unit_id]
    fig = px.line(
        unit_df,
        x="cycle",
        y=rul_col,
        title=f"RUL Trend — Unit {unit_id}",
        labels={"cycle": "Operating Cycle", rul_col: "Remaining Useful Life"},
    )
    fig.update_layout(template="plotly_dark")
    return fig


def plot_sensor_timeseries(
    df: pd.DataFrame, unit_id: int, sensors: list[str] | None = None
) -> go.Figure:
    """Plot sensor readings over time for a single asset."""
    unit_df = df[df["unit_id"] == unit_id]
    sensors = sensors or [f"sensor_{i}" for i in range(1, 6)]
    fig = go.Figure()
    for sensor in sensors:
        if sensor in unit_df.columns:
            fig.add_trace(
                go.Scatter(x=unit_df["cycle"], y=unit_df[sensor], name=sensor, mode="lines")
            )
    fig.update_layout(
        title=f"Sensor Readings — Unit {unit_id}",
        xaxis_title="Cycle",
        yaxis_title="Sensor Value",
        template="plotly_dark",
    )
    return fig


def plot_survival_curve(curve_df: pd.DataFrame, asset_id: str) -> go.Figure:
    """Conditional survival probability from current cycle (Cox PH)."""
    fig = px.line(
        curve_df,
        x="cycle",
        y="survival_prob",
        title=f"Survival curve — {asset_id}",
        labels={
            "cycle": "Future operating cycle",
            "survival_prob": "P(operational | survived to current cycle)",
        },
    )
    fig.update_layout(
        template="plotly_dark",
        yaxis=dict(range=[0, 1.05]),
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="50%")
    return fig


def plot_health_gauge(score: float, title: str = "Health Score") -> go.Figure:
    """Create a gauge chart for asset health score (0-100)."""
    color = "green" if score >= 70 else "orange" if score >= 40 else "red"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#ffcccc"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d4edda"},
                ],
            },
        )
    )
    fig.update_layout(height=300, template="plotly_dark")
    return fig


def plot_confusion_matrix_heatmap(cm, labels: list[str] | None = None) -> plt.Figure:
    """Matplotlib heatmap for classification confusion matrix."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig
