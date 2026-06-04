"""Active alerts from Phase 3 fleet predictions (UC5 Component C)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pandas as pd
import streamlit as st

from dashboard.data_loader import load_fleet_predictions, render_dataset_selector
from dashboard.page_init import init_page
from src.alerts.alert_payload import fleet_row_to_alert
from src.alerts.cmms_mock import CMMSClient

st.set_page_config(page_title="Active Alerts", layout="wide")
init_page()
st.title("Active Alerts")

dataset_id = render_dataset_selector()
fleet = load_fleet_predictions(dataset_id)
if fleet is None:
    st.warning(
        f"Run Phase 3 for {dataset_id}: "
        f"`python scripts/train_cmapss_phase3.py --dataset {dataset_id}` then "
        f"`python scripts/export_fleet_predictions.py --dataset {dataset_id}`"
    )
    st.stop()

if "sensor_readings_json" not in fleet.columns:
    st.info(
        "Re-export predictions for full UC5 alert fields: "
        f"`python scripts/export_fleet_predictions.py --dataset {dataset_id}`"
    )

alerts_df = fleet[fleet["alert_level"].isin(["warning", "critical"])].copy()
level_filter = st.multiselect(
    "Filter by Level", ["critical", "warning"], default=["critical", "warning"]
)
filtered = alerts_df[alerts_df["alert_level"].isin(level_filter)]

st.caption(
    "Each alert includes asset ID, sensor snapshot, risk score, time to failure (cycles), "
    "and recommended maintenance action per UC5 Component C."
)

display_cols = [
    "asset_id",
    "alert_level",
    "escalation_tier",
    "risk_score",
    "time_to_failure_cycles",
    "failure_prob_30",
    "anomaly_score",
    "recommended_action",
]
display_cols = [c for c in display_cols if c in filtered.columns]
st.dataframe(
    filtered[display_cols],
    use_container_width=True,
    hide_index=True,
)

for _, row in filtered.iterrows():
    with st.expander(f"{row['asset_id']} — {str(row['alert_level']).upper()}"):
        st.markdown(f"**Risk score:** {row.get('risk_score', row.get('health_score', 0)):.1f} / 100")
        ttf = row.get("time_to_failure_cycles", row.get("rul_pred", 0))
        st.markdown(f"**Time to predicted failure:** {ttf:.0f} cycles")
        st.markdown(f"**Recommended action:** {row.get('recommended_action', '—')}")
        if "sensor_readings_json" in row.index and pd.notna(row["sensor_readings_json"]):
            sensors = json.loads(str(row["sensor_readings_json"]))
            st.markdown("**Current sensor readings (last cycle):**")
            st.json(sensors)
        st.markdown(f"**Alert message:** {row.get('alert_message', '')}")

with st.expander("CMMS integration (deployment design)"):
    st.markdown(
        """
In production, the alert pipeline would:

1. **Emit** a work-order payload when `alert_level` is warning or critical (webhook or message queue).
2. **POST** to the CMMS REST API (e.g. SAP PM, Maximo) with asset ID, priority, recommended action, sensor snapshot, RUL, and risk score.
3. **Map** `escalation_tier` → CMMS priority / SLA (L2-Critical = 4h response, L1-Warning = 72h).
4. **Acknowledge** in CMMS → update alert status; unacknowledged critical alerts **escalate** to supervisor after SLA breach.
5. **Close the loop** when maintenance is completed (work order status feeds back to the dashboard).

The button below uses `CMMSClient` with a mock endpoint; if the API is unreachable, payloads are logged locally.
        """
    )

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Submit work orders to CMMS (mock)") and len(filtered) > 0:
        cmms = CMMSClient()
        results = []
        for i, (_, row) in enumerate(filtered.iterrows(), start=1):
            alert = fleet_row_to_alert(row, alert_id=f"ALT-{i:06d}")
            if alert:
                results.append(cmms.create_work_order(alert))
        st.success(f"Processed {len(results)} work order(s).")
        with st.expander("Last CMMS payload"):
            if results:
                st.json(results[-1])

with col_b:
    st.caption("Configure `CMMS_API_URL` and `CMMS_API_KEY` in `.env` for a real endpoint.")
