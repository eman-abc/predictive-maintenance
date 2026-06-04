"""Ollama-powered Phase 3 metric explanations for Model Metrics."""

from __future__ import annotations

import os

import httpx
import streamlit as st

from dashboard.api_client import post_metric_explain, use_api_backend
from dashboard.ollama_startup import get_ollama_client
from src.briefings.metric_explanation_prompts import (
    SECTION_ANOMALY,
    SECTION_COX,
    SECTION_FAILURE,
    SECTION_HEADLINE,
    SECTION_RUL,
    SECTION_SUMMARIZE,
    SYSTEM_PROMPT,
    build_explanation_prompt,
    build_instant_explanation,
    build_section_payload,
    build_summarizer_payload,
    build_summarizer_prompt,
)


def _cache_key(dataset_id: str, section_id: str) -> str:
    return f"metric_expl_{dataset_id}_{section_id}"


def _metrics_num_predict(section_id: str) -> int:
    if section_id == SECTION_SUMMARIZE:
        return int(os.getenv("OLLAMA_NUM_PREDICT_METRICS_SUMMARY", "450"))
    return int(os.getenv("OLLAMA_NUM_PREDICT_METRICS", "320"))


def _run_generation(
    *,
    section_id: str,
    dataset_id: str,
    summary: dict,
    registry_entry: dict | None,
) -> tuple[str, str] | None:
    """Generate explanation text. Returns (text, source) or None on failure."""
    if use_api_backend():
        try:
            with st.spinner("Explaining metrics via API…"):
                result = post_metric_explain(
                    section_id=section_id,
                    dataset_id=dataset_id,
                    summary=summary,
                    registry_entry=registry_entry,
                )
            return result["text"], result.get("source", "ai")
        except httpx.TimeoutException:
            text = build_instant_explanation(section_id, summary, dataset_id)
            st.warning("API timed out — showing instant summary instead.")
            return text, "instant_timeout"
        except Exception as exc:
            st.error(f"Could not generate explanation: {exc}")
            return None

    client = get_ollama_client()
    if not client.is_available():
        text = build_instant_explanation(section_id, summary, dataset_id)
        return text, "instant"

    if section_id == SECTION_SUMMARIZE:
        payload = build_summarizer_payload(summary, dataset_id, registry_entry=registry_entry)
        prompt = build_summarizer_prompt(payload)
    else:
        payload = build_section_payload(
            section_id, summary, dataset_id, registry_entry=registry_entry
        )
        prompt = build_explanation_prompt(section_id, payload)

    num_predict = _metrics_num_predict(section_id)
    try:
        with st.spinner("Explaining metrics…"):
            text = client.generate(
                prompt,
                system=SYSTEM_PROMPT,
                num_predict=num_predict,
            )
    except httpx.TimeoutException:
        st.error(
            "Timed out — click **Explain** again, use a smaller model, or raise `OLLAMA_TIMEOUT` in `.env`."
        )
        return None
    except Exception as exc:
        st.error(f"Could not generate explanation: {exc}")
        return None

    return text.strip(), "ai"


def _store_result(dataset_id: str, section_id: str, text: str) -> None:
    st.session_state[_cache_key(dataset_id, section_id)] = text


def _render_cached(dataset_id: str, section_id: str) -> None:
    cached = st.session_state.get(_cache_key(dataset_id, section_id))
    if not cached:
        return
    st.info(cached)


def render_metric_explanation_controls(
    section_id: str,
    dataset_id: str,
    summary: dict,
    *,
    registry_entry: dict | None = None,
    button_prefix: str = "",
) -> None:
    """Single Explain button; each click replaces the previous explanation."""
    prefix = button_prefix or section_id
    if st.button(
        "Explain",
        key=f"metric_explain_{prefix}_{dataset_id}",
        help="Generate a fresh Ollama interpretation of this section's metrics.",
    ):
        result = _run_generation(
            section_id=section_id,
            dataset_id=dataset_id,
            summary=summary,
            registry_entry=registry_entry,
        )
        if result:
            _store_result(dataset_id, section_id, result[0])

    _render_cached(dataset_id, section_id)


def render_summarize_all_controls(
    dataset_id: str,
    summary: dict,
    *,
    registry_entry: dict | None = None,
) -> None:
    """Executive summary across all Phase 3 metrics for the current FD subset."""
    st.markdown("**Executive summary (all metrics)**")
    st.caption(
        "Summarizes **this dataset only** using the same Ollama model as asset briefings. "
        "AI text is advisory — confirm against the tables above."
    )

    c_ai, c_instant = st.columns(2)
    with c_instant:
        if st.button(
            "Instant summary",
            key=f"metric_instant_summarize_{dataset_id}",
            help="Template summary (no Ollama wait).",
        ):
            text = build_instant_explanation(SECTION_SUMMARIZE, summary, dataset_id)
            _store_result(dataset_id, SECTION_SUMMARIZE, text)
            st.rerun()
    with c_ai:
        render_metric_explanation_controls(
            SECTION_SUMMARIZE,
            dataset_id,
            summary,
            registry_entry=registry_entry,
            button_prefix="summarize",
        )


__all__ = [
    "SECTION_ANOMALY",
    "SECTION_COX",
    "SECTION_FAILURE",
    "SECTION_HEADLINE",
    "SECTION_RUL",
    "render_metric_explanation_controls",
    "render_summarize_all_controls",
]
