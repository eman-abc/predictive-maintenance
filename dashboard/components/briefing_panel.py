"""Ollama-powered maintenance briefing for a single fleet asset."""

from __future__ import annotations

import json

import httpx
import pandas as pd
import streamlit as st

from src.briefings.ollama_client import OllamaClient
from src.briefings.prompt_templates import SYSTEM_PROMPT, build_briefing_prompt


def _sensor_summary_from_row(row: pd.Series) -> dict[str, float] | None:
    if "sensor_readings_json" not in row.index or pd.isna(row["sensor_readings_json"]):
        return None
    try:
        parsed = json.loads(str(row["sensor_readings_json"]))
    except json.JSONDecodeError:
        return None
    return {str(k): float(v) for k, v in parsed.items()}


def _briefing_cache_key(dataset_id: str, asset_id: str) -> str:
    return f"briefing_{dataset_id}_{asset_id}"


def render_asset_briefing(row: pd.Series, *, dataset_id: str) -> None:
    """Show UC5 Component D: generate and display a per-asset LLM briefing."""
    st.subheader("AI Maintenance Briefing")
    asset_id = str(row["asset_id"])
    client = OllamaClient()

    if not client.is_available():
        st.warning(
            "Ollama is not reachable. Start the local server and pull a model:\n\n"
            "```bash\nollama pull llama3.2\nollama serve\n```\n\n"
            "Then set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env` (see `.env.example`)."
        )
        return

    st.caption(f"Model: **{client.model}** · Server: `{client.base_url}`")
    st.caption(
        "First run can take **2–5 minutes** while Ollama loads the model "
        "(longer on battery power). Later requests are much faster."
    )

    warmup_key = f"ollama_warm_{client.model}"
    col_gen, col_warm = st.columns([1, 1])
    with col_gen:
        generate = st.button(
            "Generate briefing",
            type="primary",
            key=f"brief_btn_{dataset_id}_{asset_id}",
        )
    with col_warm:
        preload = st.button(
            "Preload model",
            key=f"warm_btn_{dataset_id}_{asset_id}",
            help="Load the model into memory before generating a full briefing.",
        )

    if preload:
        with st.spinner("Loading model into memory — please wait…"):
            try:
                client.warmup()
                st.session_state[warmup_key] = True
                st.success("Model loaded. Click **Generate briefing**.")
            except Exception as exc:
                st.error(f"Preload failed: {exc}")

    if generate:
        failure_prob = float(row.get("failure_prob_30", row.get("failure_prob", 0)))
        prompt = build_briefing_prompt(
            asset_id=asset_id,
            health_score=float(row["health_score"]),
            rul=float(row["rul_pred"]),
            failure_probability=failure_prob,
            sensor_summary=_sensor_summary_from_row(row),
        )
        with st.spinner(
            "Generating briefing — first run may take several minutes…"
        ):
            try:
                text = client.generate(prompt, system=SYSTEM_PROMPT)
            except httpx.TimeoutException:
                st.error(
                    "Request timed out. Click **Preload model**, wait for it to finish, "
                    "then try again. You can also raise `OLLAMA_TIMEOUT` in `.env` (default 300s)."
                )
                return
            except Exception as exc:
                st.error(f"Could not generate briefing: {exc}")
                return
        st.session_state[_briefing_cache_key(dataset_id, asset_id)] = text

    cached = st.session_state.get(_briefing_cache_key(dataset_id, asset_id))
    if cached:
        st.info(cached)
