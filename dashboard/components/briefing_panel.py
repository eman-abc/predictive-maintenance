"""Ollama-powered maintenance briefing for a single fleet asset."""

from __future__ import annotations

import json

import httpx
import pandas as pd
import streamlit as st

from dashboard.briefing_api import (
    SYSTEM_PROMPT,
    build_briefing_prompt,
    build_instant_briefing,
)
from dashboard.ollama_startup import get_ollama_client


def _sensor_summary_from_row(row: pd.Series, *, max_sensors: int = 4) -> dict[str, float] | None:
    if "sensor_readings_json" not in row.index or pd.isna(row["sensor_readings_json"]):
        return None
    try:
        parsed = json.loads(str(row["sensor_readings_json"]))
    except json.JSONDecodeError:
        return None
    items = list(parsed.items())[:max_sensors]
    return {str(k): float(v) for k, v in items}


def _briefing_cache_key(dataset_id: str, asset_id: str) -> str:
    return f"briefing_{dataset_id}_{asset_id}"


def _row_context(row: pd.Series, asset_id: str) -> dict:
    failure_prob = float(row.get("failure_prob_30", row.get("failure_prob", 0)))
    return {
        "asset_id": asset_id,
        "health_score": float(row["health_score"]),
        "rul": float(row["rul_pred"]),
        "failure_probability": failure_prob,
        "alert_level": str(row.get("alert_level", "normal")),
        "recommended_action": str(row.get("recommended_action", "")),
        "anomaly_score": float(row.get("anomaly_score", 0)),
        "sensor_summary": _sensor_summary_from_row(row),
    }


def render_asset_briefing(row: pd.Series, *, dataset_id: str) -> None:
    """Show UC5 Component D: instant template or streamed Ollama briefing."""
    st.subheader("AI Maintenance Briefing")
    asset_id = str(row["asset_id"])
    ctx = _row_context(row, asset_id)
    client = get_ollama_client()
    preload_status = st.session_state.get("ollama_preload_status", "")

    if not client.is_available():
        st.warning(
            "Ollama is not reachable. Use **Instant briefing** below, or start Ollama:\n\n"
            "```bash\nollama pull llama3.2\n```\n\n"
            "Set `OLLAMA_MODEL=llama3.2` in `.env` (faster than llama3 8B)."
        )
    else:
        ready_note = (
            "Model preloaded at startup."
            if preload_status == "ready"
            else "Model will load on first app open."
        )
        st.caption(
            f"Model: **{client.model}** · Server: `{client.base_url}` · {ready_note}"
        )

    col_instant, col_ai = st.columns(2)
    with col_instant:
        instant = st.button(
            "Instant briefing",
            key=f"instant_btn_{dataset_id}_{asset_id}",
            help="No LLM — uses model predictions only (<1s).",
        )
    with col_ai:
        generate = st.button(
            "AI briefing",
            type="primary",
            key=f"brief_btn_{dataset_id}_{asset_id}",
            disabled=not client.is_available(),
        )

    if instant:
        text = build_instant_briefing(
            asset_id=ctx["asset_id"],
            health_score=ctx["health_score"],
            rul=ctx["rul"],
            failure_probability=ctx["failure_probability"],
            alert_level=ctx["alert_level"],
            recommended_action=ctx["recommended_action"],
            anomaly_score=ctx["anomaly_score"],
        )
        st.session_state[_briefing_cache_key(dataset_id, asset_id)] = text
        st.session_state[f"{_briefing_cache_key(dataset_id, asset_id)}_source"] = "instant"

    if generate and client.is_available():
        prompt = build_briefing_prompt(
            ctx["asset_id"],
            ctx["health_score"],
            ctx["rul"],
            ctx["failure_probability"],
            sensor_summary=ctx["sensor_summary"],
        )
        placeholder = st.empty()
        parts: list[str] = []
        try:
            with st.spinner("Generating…"):
                stream_fn = getattr(client, "generate_stream", None)
                if callable(stream_fn):
                    for token in stream_fn(prompt, system=SYSTEM_PROMPT):
                        parts.append(token)
                        placeholder.markdown("".join(parts))
                    text = "".join(parts)
                else:
                    text = client.generate(prompt, system=SYSTEM_PROMPT)
                    placeholder.markdown(text)
        except httpx.TimeoutException:
            st.error(
                "Timed out — use **Instant briefing** or set `OLLAMA_NUM_PREDICT=80` in `.env`, "
                "then restart the app."
            )
            return
        except Exception as exc:
            st.error(f"Could not generate briefing: {exc}")
            return
        st.session_state[_briefing_cache_key(dataset_id, asset_id)] = text
        st.session_state[f"{_briefing_cache_key(dataset_id, asset_id)}_source"] = "ai"

    cached = st.session_state.get(_briefing_cache_key(dataset_id, asset_id))
    if cached:
        source = st.session_state.get(
            f"{_briefing_cache_key(dataset_id, asset_id)}_source", "ai"
        )
        label = "Instant" if source == "instant" else "AI"
        st.info(f"**{label} briefing:** {cached}")
