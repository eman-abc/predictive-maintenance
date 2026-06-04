"""Active alerts from Phase 3 fleet predictions (UC5 Component C)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import pandas as pd
import streamlit as st

from dashboard.api_client import (
    get_cmms_databricks_status,
    get_recent_auto_work_orders,
    get_recent_work_orders,
    post_alert_ack,
    post_auto_dispatch,
    post_work_orders,
    use_api_backend,
)
from dashboard.components.shift_briefing_panel import render_shift_briefing
from dashboard.data_loader import load_fleet_predictions, render_dataset_selector
from dashboard.page_init import init_page
from dashboard.theme import alert_level_color, style_fleet_dataframe
from src.alerts.cmms_databricks import (
    databricks_status_payload,
    fetch_recent_auto_work_orders,
    fetch_recent_work_orders,
    is_databricks_logging_configured,
)
from src.alerts.cmms_routing import map_escalation
from src.alerts.cmms_mock import CMMSClient
from src.services.cmms_auto_dispatch import auto_dispatch_critical

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

level_filter = st.multiselect(
    "Filter by Level", ["critical", "warning"], default=["critical", "warning"]
)

if use_api_backend():
    from dashboard.api_client import get_alerts_df

    filtered = get_alerts_df(dataset_id, level_filter)
else:
    alerts_df = fleet[fleet["alert_level"].isin(["warning", "critical"])].copy()
    filtered = alerts_df[alerts_df["alert_level"].isin(level_filter)].copy()
    if "ack_status" not in filtered.columns:
        filtered["ack_status"] = "open"
    for idx, row in filtered.iterrows():
        ack_key = f"ack_{dataset_id}_{row['asset_id']}_{row['alert_level']}"
        if st.session_state.get(ack_key):
            filtered.at[idx, "ack_status"] = "acknowledged"

render_shift_briefing(dataset_id=dataset_id, levels=level_filter)

st.caption(
    "Each alert includes asset ID, sensor snapshot, risk score, time to failure (cycles), "
    "and recommended maintenance action per UC5 Component C."
)

display_cols = [
    "asset_id",
    "alert_level",
    "ack_status",
    "escalation_tier",
    "risk_score",
    "time_to_failure_cycles",
    "failure_prob_30",
    "anomaly_score",
    "recommended_action",
]
display_cols = [c for c in display_cols if c in filtered.columns]
styled = filtered[display_cols].copy()
if "alert_level" in styled.columns:
    styled = styled.rename(columns={"alert_level": "status"})
st.dataframe(
    style_fleet_dataframe(styled),
    use_container_width=True,
    hide_index=True,
)

# Two-phase auto-dispatch: first paint shows the table; second run talks to Databricks.
_auto_key = f"auto_dispatch_done_{dataset_id}"
_pending_key = f"auto_dispatch_pending_{dataset_id}"
if _auto_key not in st.session_state:
    if not st.session_state.get(_pending_key):
        st.session_state[_pending_key] = True
        st.rerun()
    else:
        del st.session_state[_pending_key]
        with st.spinner("Auto-dispatching critical alerts to Databricks…"):
            try:
                if use_api_backend():
                    auto_result = post_auto_dispatch(dataset_id, ["critical"])
                else:
                    auto_result = auto_dispatch_critical(dataset_id, levels=["critical"])
            except Exception as exc:
                auto_result = {"status": "error", "reason": str(exc), "dispatched_count": 0}
        st.session_state[_auto_key] = auto_result
        st.rerun()
else:
    auto_result = st.session_state[_auto_key]
    if auto_result.get("dispatched_count", 0) > 0:
        st.success(
            f"Auto-dispatched **{auto_result['dispatched_count']}** critical work order(s) "
            f"to `{auto_result.get('table', 'cmms_work_orders_auto')}`."
        )
    elif auto_result.get("status") == "skipped":
        st.caption(f"Auto-dispatch skipped: {auto_result.get('reason', '')}")
    elif auto_result.get("status") == "error":
        st.warning(
            f"Auto-dispatch failed: {auto_result.get('reason', 'unknown')}. "
            "Alerts above are still valid; check Databricks SQL settings in `.env`."
        )
        if st.button("Retry auto-dispatch", key=f"retry_auto_{dataset_id}"):
            del st.session_state[_auto_key]
            st.rerun()

for _, row in filtered.iterrows():
    level = str(row.get("alert_level", row.get("status", "warning"))).lower()
    asset_id = str(row["asset_id"])
    ack = str(row.get("ack_status", "open"))
    with st.expander(f"{asset_id} — {level.upper()} ({ack})"):
        st.markdown(
            f"Level: <span style='color:{alert_level_color(level)};font-weight:700'>"
            f"{level.upper()}</span> · Ack: **{ack}**",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Risk score:** {row.get('risk_score', row.get('health_score', 0)):.1f} / 100")
        ttf = row.get("time_to_failure_cycles", row.get("rul_pred", 0))
        st.markdown(f"**Time to predicted failure:** {ttf:.0f} cycles")
        st.markdown(f"**Recommended action:** {row.get('recommended_action', '—')}")
        if "sensor_readings_json" in row.index and pd.notna(row["sensor_readings_json"]):
            sensors = json.loads(str(row["sensor_readings_json"]))
            st.markdown("**Current sensor readings (last cycle):**")
            st.json(sensors)
        st.markdown(f"**Alert message:** {row.get('alert_message', '')}")
        if ack != "acknowledged":
            if st.button("Acknowledge alert", key=f"ack_{dataset_id}_{asset_id}_{level}"):
                if use_api_backend():
                    post_alert_ack(dataset_id, asset_id, level)
                else:
                    st.session_state[f"ack_{dataset_id}_{asset_id}_{level}"] = True
                st.rerun()

def _db_status() -> dict:
    if use_api_backend():
        try:
            return get_cmms_databricks_status()
        except Exception:
            return {"configured": False}
    if is_databricks_logging_configured():
        return databricks_status_payload()
    return {"configured": False}


db = _db_status()
if db.get("configured"):
    st.caption(
        f"Manual CMMS log: `{db.get('table_fqn')}` · "
        f"Auto-dispatch: `{db.get('auto_table_fqn')}`"
        + (f" · [Manual table]({db.get('explore_url')})" if db.get("explore_url") else "")
        + (f" · [Auto table]({db.get('auto_explore_url')})" if db.get("auto_explore_url") else "")
    )
else:
    st.caption(
        "Set `CMMS_LOG_TO_DATABRICKS=true` and Databricks SQL vars in `.env`, then "
        "`python scripts/setup_cmms_databricks_table.py`."
    )

with st.expander("CMMS integration (deployment design)"):
    st.markdown(
        """
- **Auto-dispatch:** critical alerts are written to `cmms_work_orders_auto` when this page loads (pipeline simulation).
- **Manual submit:** operator button logs warning + critical to `cmms_work_orders`.
- **Escalation:** L2-Critical → P1 / 4h SLA; L1-Warning → P2 / 72h.
- **Acknowledge:** mock operator ack (in-memory on API for demo).
        """
    )

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Submit work orders to CMMS (manual)") and len(filtered) > 0:
        with st.spinner("Submitting work orders…"):
            if use_api_backend():
                batch = post_work_orders(dataset_id, list(level_filter))
                results = batch.get("results") or []
            else:
                cmms = CMMSClient()
                results = []
                for i, (_, row) in enumerate(filtered.iterrows(), start=1):
                    from src.alerts.alert_payload import fleet_row_to_alert

                    alert = fleet_row_to_alert(row, alert_id=f"ALT-{i:06d}")
                    if alert:
                        results.append(
                            cmms.create_work_order(alert, dataset_id=dataset_id)
                        )
        logged_db = sum(
            1 for r in results if (r.get("databricks") or {}).get("status") == "databricks_logged"
        )
        st.success(
            f"Processed {len(results)} work order(s)."
            + (f" {logged_db} row(s) written to Databricks (manual table)." if logged_db else "")
        )
        if results:
            payload = results[-1].get("payload") or {}
            routing = map_escalation(payload.get("escalation_tier"))
            st.markdown(
                f"**CMMS routing:** `{routing.escalation_tier}` → **{routing.cmms_priority}** "
                f"({routing.sla_label})"
            )
            db_info = results[-1].get("databricks") or {}
            if db_info.get("explore_url"):
                st.link_button("Open manual table in Catalog Explorer", db_info["explore_url"])

with col_b:
    st.caption("Auto table = pipeline; manual table = operator button.")

if db.get("configured"):
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Recent manual work orders", expanded=False):
            try:
                rows = (
                    get_recent_work_orders(15)
                    if use_api_backend()
                    else fetch_recent_work_orders(limit=15)
                )
                st.dataframe(rows, use_container_width=True, hide_index=True) if rows else st.info("Empty")
            except Exception as exc:
                st.warning(str(exc))
    with c2:
        with st.expander("Recent auto-dispatched (critical)", expanded=False):
            try:
                rows = (
                    get_recent_auto_work_orders(15)
                    if use_api_backend()
                    else fetch_recent_auto_work_orders(limit=15)
                )
                st.dataframe(rows, use_container_width=True, hide_index=True) if rows else st.info("Empty")
            except Exception as exc:
                st.warning(str(exc))
