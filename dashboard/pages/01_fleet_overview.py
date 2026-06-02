"""Fleet overview — health scores from Phase 3 test predictions."""

import streamlit as st

from dashboard.data_loader import (
    load_fleet_predictions,
    load_phase3_summary,
    render_dataset_selector,
)
from dashboard.page_init import init_page

st.set_page_config(page_title="Fleet Overview", layout="wide")
init_page()
st.title("Fleet Overview")

dataset_id = render_dataset_selector()
fleet = load_fleet_predictions(dataset_id)
summary = load_phase3_summary(dataset_id)

if fleet is None:
    st.warning(
        f"No predictions for **{dataset_id}**. Run: "
        f"`python scripts/train_cmapss_phase3.py --dataset {dataset_id}`"
    )
    st.stop()

if summary:
    st.caption(
        f"**{dataset_id}** — RUL: **{summary['winner'].upper()}** | "
        f"NASA: **{summary['test_metrics']['rul_score']:.2f}** | "
        f"RMSE: **{summary['test_metrics']['rmse']:.2f}** cycles"
    )

df = fleet.rename(
    columns={
        "rul_pred": "rul",
        "alert_level": "status",
    }
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Health", f"{df['health_score'].mean():.1f}%")
col2.metric("Critical Assets", int((df["status"] == "critical").sum()))
col3.metric("Warning Assets", int((df["status"] == "warning").sum()))
if "anomaly_score" in df.columns:
    col4.metric("Avg Anomaly Score", f"{df['anomaly_score'].mean():.1f}")
    col4.caption("Higher = more unusual vs healthy train baseline")
else:
    col4.metric("Anomaly", "—")
    col4.caption("Re-export predictions with anomaly model")

display_cols = [
    "asset_id",
    "health_score",
    "anomaly_score",
    "is_anomaly",
    "rul",
    "failure_prob_30",
    "failure_prob_72",
    "status",
    "rul_true",
]
display_cols = [c for c in display_cols if c in df.columns]
st.dataframe(
    df[display_cols].style.background_gradient(subset=["health_score"], cmap="RdYlGn"),
    use_container_width=True,
    hide_index=True,
)

st.bar_chart(df.set_index("asset_id")["health_score"])
