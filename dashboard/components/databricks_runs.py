"""Streamlit panel: Databricks MLflow experiment links and live run metrics."""

from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd
import streamlit as st

from dashboard import mlflow_links as links


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(v) or math.isnan(v):
        return "—"
    return f"{v:.{digits}f}"


def _rows_to_display(rows: list[dict[str, Any]]) -> pd.DataFrame:
    display = []
    for row in rows:
        display.append(
            {
                "dataset": row.get("dataset", ""),
                "run_name": row.get("run_name", ""),
                "winner": (row.get("winner") or "—").upper()
                if row.get("winner")
                else "—",
                "test_rmse": _fmt(row.get("test_rmse")),
                "test_nasa": _fmt(row.get("test_nasa")),
                "cox_rmse": _fmt(row.get("test_cox_rmse")),
                "concordance": _fmt(row.get("cox_concordance"), 3),
                "batch": row.get("training_batch") or "—",
                "Open run": row.get("run_url") or "",
                "Artifacts": row.get("artifacts_url") or "",
            }
        )
    return pd.DataFrame(display)


def should_show_databricks_panel(registry: dict | None) -> bool:
    if links.has_databricks_credentials():
        return True
    return links.can_build_registry_links(registry)


def render_databricks_mlflow_panel(
    registry: dict | None,
    *,
    expanded: bool = True,
    title: str = "Databricks MLflow experiment",
) -> None:
    """Show experiment link, Phase 3 summary runs, and pickle bundle run."""
    if not should_show_databricks_panel(registry):
        return

    with st.expander(title, expanded=expanded):
        try:
            data = links.collect_databricks_run_links(registry)
        except Exception as exc:
            st.error(f"Databricks panel failed: {exc}")
            return

        host = links.databricks_host()
        if host:
            st.caption(
                f"Workspace: `{host}` · experiment `{data['experiment_name']}`"
                + (f" · id `{data['experiment_id']}`" if data.get("experiment_id") else "")
            )

        if data.get("registry_only"):
            st.info(
                "Run links from local registry (no API). Metrics from "
                "`cmapss_training_registry.json`."
            )
        elif not links.has_databricks_credentials() and links.resolve_experiment_id(registry):
            st.caption(
                "Links use `MLFLOW_EXPERIMENT_ID` or `registry.mlflow_experiment_id` + "
                "per-dataset `mlflow_run_id`."
            )

        if data.get("error"):
            st.warning(f"Live MLflow query failed: {data['error']}. Using registry fallback.")

        if data.get("experiment_url"):
            st.link_button("Open experiment in Databricks", data["experiment_url"])

        phase3 = data.get("phase3_rows") or []
        if phase3:
            source = data.get("source", "none")
            st.markdown(f"**Phase 3 summary runs** ({len(phase3)})")
            if source.startswith("registry"):
                st.caption("Source: local registry + experiment id.")
            elif source == "live":
                st.caption("Source: live MLflow API.")
            df = _rows_to_display(phase3)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Open run": st.column_config.LinkColumn(
                        "Open run",
                        display_text="View run ↗",
                    ),
                    "Artifacts": st.column_config.LinkColumn(
                        "Artifacts",
                        display_text="models/ ↗",
                    ),
                },
            )

        bundles = data.get("bundle_rows") or []
        if bundles:
            st.markdown("**Model artifact bundle** (`.pkl` uploads)")
            for bundle in bundles:
                cols = st.columns([3, 1, 1])
                cols[0].markdown(f"`{bundle['run_name']}`")
                if bundle.get("run_url"):
                    cols[1].link_button("View run", bundle["run_url"], use_container_width=True)
                if bundle.get("artifacts_url"):
                    cols[2].link_button("models/", bundle["artifacts_url"], use_container_width=True)

        if not phase3 and not bundles:
            st.info(
                "No run links yet. Check `artifacts/cmapss_training_registry.json` and "
                "`MLFLOW_EXPERIMENT_ID` in `.env`."
            )
