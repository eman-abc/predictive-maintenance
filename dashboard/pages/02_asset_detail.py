"""Per-asset deep-dive using test trajectory + Phase 3 predictions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.components.briefing_panel import render_asset_briefing
from dashboard.page_init import init_page
from dashboard.components.health_gauge import render_health_gauge
from dashboard.components.rul_chart import render_rul_chart
from dashboard.components.sensor_chart import render_sensor_chart
from dashboard.components.survival_chart import render_survival_chart
from dashboard.data_loader import (
    load_fleet_predictions,
    load_survival_model,
    load_unit_trajectory,
    render_dataset_selector,
)

def _optional_metric(row, key: str) -> float | None:
    if key not in row.index or pd.isna(row[key]):
        return None
    return float(row[key])


def _display_rul(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    v = float(value)
    if v == float("inf") or v == float("-inf"):
        return "—"
    return f"{v:.0f}"


st.set_page_config(page_title="Asset Detail", layout="wide")
init_page()
st.title("Asset Detail")

dataset_id = render_dataset_selector()
fleet = load_fleet_predictions(dataset_id)
if fleet is None:
    st.warning(
        f"Run Phase 3 for {dataset_id}: "
        f"`python scripts/train_cmapss_phase3.py --dataset {dataset_id}`"
    )
    st.stop()

assets = sorted(fleet["asset_id"].tolist())
asset_id = st.selectbox("Select Asset", assets)
unit_id = int(asset_id.split("-")[1])

row = fleet[fleet["asset_id"] == asset_id].iloc[0]
trajectory = load_unit_trajectory(unit_id, dataset_id)

col1, col2 = st.columns([1, 2])
with col1:
    render_health_gauge(float(row["health_score"]), title=f"{asset_id} Health")
    st.metric("Predicted RUL (winner)", f"{row['rul_pred']:.0f} cycles")
    if "rul_pred_cox" in row.index:
        st.metric("Cox survival RUL", f"{_display_rul(row['rul_pred_cox'])} cycles")
    st.metric("True RUL (last cycle)", f"{row['rul_true']:.0f} cycles")
    st.metric("P(failure ≤30 cycles)", f"{row.get('failure_prob_30', row['failure_prob']):.0%}")
    if "failure_prob_72" in row.index:
        st.metric("P(failure ≤72 cycles)", f"{row['failure_prob_72']:.0%}")
    if "survival_prob_30" in row.index and pd.notna(row["survival_prob_30"]):
        st.metric("P(survive 30+ cycles)", f"{row['survival_prob_30']:.0%}")
    if "anomaly_score" in row.index:
        st.metric("Anomaly score", f"{row['anomaly_score']:.1f}")
        st.caption(
            "Isolation Forest vs healthy training cycles (RUL≥30). "
            f"Flagged: {'yes' if row.get('is_anomaly', 0) else 'no'}"
        )
    st.metric("Alert Level", row["alert_level"].upper())

with col2:
    if trajectory is not None and "rul" in trajectory.columns:
        trend_df = trajectory[["cycle", "rul"]].copy()
        trend_df["rul_predicted_endpoint"] = float(row["rul_pred"])
        render_rul_chart(trend_df, asset_id)
    else:
        st.info("Trajectory data unavailable.")

if trajectory is not None:
    sensor_cols = [c for c in trajectory.columns if c.startswith("sensor_")][:3]
    if sensor_cols:
        sensor_df = trajectory[["cycle"] + sensor_cols]
        render_sensor_chart(sensor_df, asset_id)

survival_model = load_survival_model(dataset_id)
if survival_model is not None and trajectory is not None and len(trajectory) > 0:
    st.subheader("Cox survival analysis")
    last_cycle_row = trajectory.iloc[-1]
    curve_df = survival_model.survival_curve(
        last_cycle_row,
        current_cycle=float(last_cycle_row["cycle"]),
    )
    render_survival_chart(
        curve_df,
        asset_id,
        rul_pred_cox=_optional_metric(row, "rul_pred_cox"),
        survival_prob_30=_optional_metric(row, "survival_prob_30"),
    )

st.divider()
render_asset_briefing(row, dataset_id=dataset_id)
