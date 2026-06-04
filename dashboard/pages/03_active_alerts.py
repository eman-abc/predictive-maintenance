"""Active alerts from Phase 3 fleet predictions (UC5 Component C)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pandas as pd
import streamlit as st

from dashboard.data_loader import load_fleet_predictions, render_dataset_selector
from dashboard.page_init import init_page
from dashboard.theme import alert_level_color, style_fleet_dataframe
from dashboard.api_client import (
    get_cmms_databricks_status,
    get_recent_work_orders,
    post_work_orders,
    use_api_backend,
)
from src.alerts.cmms_databricks import (
    databricks_table_links,
    fetch_recent_work_orders,
    is_databricks_logging_configured,
    table_fqn,
)
from src.alerts.cmms_routing import map_escalation
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
styled = filtered[display_cols].copy()
if "alert_level" in styled.columns:
    styled = styled.rename(columns={"alert_level": "status"})
st.dataframe(
    style_fleet_dataframe(styled),
    use_container_width=True,
    hide_index=True,
)

for _, row in filtered.iterrows():
    level = str(row["alert_level"]).lower()
    with st.expander(f"{row['asset_id']} — {level.upper()}"):
        st.markdown(
            f"Level: <span style='color:{alert_level_color(level)};font-weight:700'>"
            f"{level.upper()}</span>",
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

with st.expander("CMMS integration (deployment design)"):
    st.markdown(
        """
In production, the alert pipeline would:

1. **Emit** a work-order payload when `alert_level` is warning or critical (webhook or message queue).
2. **POST** to the CMMS REST API (e.g. SAP PM, Maximo) with asset ID, priority, recommended action, sensor snapshot, RUL, and risk score.
3. **Map** `escalation_tier` → CMMS priority / SLA (L2-Critical = 4h response, L1-Warning = 72h).
4. **Acknowledge** in CMMS → update alert status; unacknowledged critical alerts **escalate** to supervisor after SLA breach.
5. **Close the loop** when maintenance is completed (work order status feeds back to the dashboard).

The button below uses `CMMSClient` with a mock REST endpoint. When `CMMS_LOG_TO_DATABRICKS=true`, rows are also written to a **Delta table** in Databricks for interview evidence.
        """
    )

def _databricks_configured() -> bool:
    if use_api_backend():
        try:
            return bool(get_cmms_databricks_status().get("configured"))
        except Exception:
            return False
    return is_databricks_logging_configured()


if _databricks_configured():
    links = (
        get_cmms_databricks_status()
        if use_api_backend()
        else databricks_table_links()
    )
    fqn = links.get("table_fqn") or (table_fqn() if not use_api_backend() else "—")
    explore = links.get("explore_url")
    st.caption(
        f"Databricks audit log: `{fqn}`"
        + (f" · [Catalog Explorer]({explore})" if explore else "")
        + (f" · [SQL warehouse]({links.get('sql_editor_url')})" if links.get("sql_editor_url") else "")
    )
else:
    st.caption(
        "To log work orders to Databricks: set `CMMS_LOG_TO_DATABRICKS=true`, "
        "`CMMS_DELTA_TABLE`, `DATABRICKS_SQL_WAREHOUSE_ID`, then run "
        "`python scripts/setup_cmms_databricks_table.py`."
    )

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Submit work orders to CMMS (mock)") and len(filtered) > 0:
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
            + (f" {logged_db} row(s) written to Databricks." if logged_db else "")
        )

        last = results[-1] if results else {}
        payload = last.get("payload") or {}
        routing = map_escalation(payload.get("escalation_tier"))
        st.markdown(
            f"**CMMS routing:** `{routing.escalation_tier}` → priority **{routing.cmms_priority}** "
            f"({routing.sla_label})"
        )

        db_info = last.get("databricks") or {}
        if db_info.get("status") == "databricks_logged":
            explore = db_info.get("explore_url")
            sql_url = db_info.get("sql_editor_url")
            st.markdown("**View work orders in Databricks:**")
            if explore:
                st.link_button("Open table in Catalog Explorer", explore, type="primary")
            if sql_url:
                st.link_button("Open SQL warehouse", sql_url)
            if db_info.get("sample_query"):
                st.caption("Or run in SQL editor:")
                st.code(db_info["sample_query"], language="sql")

        with st.expander("Last submission detail"):
            if results:
                st.json(results[-1])

with col_b:
    st.caption("Configure `CMMS_API_URL` for a real CMMS, or rely on Databricks audit logging.")

if _databricks_configured():
    with st.expander("Recent work orders in Databricks", expanded=False):
        try:
            rows = (
                get_recent_work_orders(20)
                if use_api_backend()
                else fetch_recent_work_orders(limit=20)
            )
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("Table is empty — submit work orders above.")
        except Exception as exc:
            st.warning(f"Could not query Databricks: {exc}")
