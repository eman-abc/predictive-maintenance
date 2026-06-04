"""Shift handover briefing — fleet-wide summary (UC5 Component D)."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import post_shift_briefing, use_api_backend
from src.briefings.briefing_prompts import build_alert_summary_prompt, build_instant_shift_briefing
from src.services.alerts_service import list_alert_rows


def _alert_dicts(dataset_id: str, levels: list[str]) -> list[dict]:
    rows = list_alert_rows(dataset_id, levels=levels)
    return [
        {
            "asset_id": r.get("asset_id"),
            "level": r.get("alert_level"),
            "alert_level": r.get("alert_level"),
            "description": r.get("recommended_action") or r.get("alert_message", ""),
        }
        for r in rows
    ]


def render_shift_briefing(*, dataset_id: str, levels: list[str]) -> None:
    st.subheader("Shift handover briefing")
    st.caption(
        "Summarizes **all active alerts** for this dataset (not one engine). "
        "Contrast: **Asset Detail** briefing is per ENG-xxx."
    )
    col1, col2 = st.columns(2)
    with col1:
        instant = st.button("Instant shift summary", key=f"shift_instant_{dataset_id}")
    with col2:
        ai = st.button("AI shift summary", key=f"shift_ai_{dataset_id}")

    cache_key = f"shift_brief_{dataset_id}"
    if instant or ai:
        mode = "instant" if instant else "ai"
        if use_api_backend():
            result = post_shift_briefing(
                mode=mode, dataset_id=dataset_id, levels=levels
            )
            st.session_state[cache_key] = result.get("text", "")
            st.session_state[f"{cache_key}_source"] = result.get("source", mode)
        else:
            alerts = _alert_dicts(dataset_id, levels)
            if mode == "instant":
                text = build_instant_shift_briefing(alerts)
            else:
                from dashboard.ollama_startup import get_ollama_client
                from src.briefings.briefing_prompts import SYSTEM_PROMPT

                client = get_ollama_client()
                if client.is_available():
                    text = client.generate(
                        build_alert_summary_prompt(alerts), system=SYSTEM_PROMPT
                    )
                else:
                    text = build_instant_shift_briefing(alerts)
            st.session_state[cache_key] = text
            st.session_state[f"{cache_key}_source"] = mode

    if cache_key in st.session_state:
        src = st.session_state.get(f"{cache_key}_source", "instant")
        st.info(f"**{src.upper()} shift briefing:**\n\n{st.session_state[cache_key]}")
