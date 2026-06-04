"""One-time Ollama model preload per Streamlit session."""

from __future__ import annotations

import os

import streamlit as st

from dashboard.briefing_api import create_ollama_client, preload_ollama_model

_STATUS_KEY = "ollama_preload_status"
_DONE_KEY = "ollama_preload_done"


def _preload_enabled() -> bool:
    return os.getenv("OLLAMA_PRELOAD_ON_STARTUP", "true").lower() in (
        "1",
        "true",
        "yes",
    )


def ensure_ollama_preloaded(*, show_spinner: bool = True) -> str:
    """
    Load the LLM into memory once per browser session.

    Safe to call from every page; subsequent calls are no-ops.
    """
    if st.session_state.get(_DONE_KEY):
        return str(st.session_state.get(_STATUS_KEY, "ready"))

    if not _preload_enabled():
        st.session_state[_DONE_KEY] = True
        st.session_state[_STATUS_KEY] = "disabled"
        return "disabled"

    client = create_ollama_client()
    if not client.is_available():
        st.session_state[_DONE_KEY] = True
        st.session_state[_STATUS_KEY] = "unavailable"
        return "unavailable"

    try:
        if show_spinner:
            with st.spinner(
                "Loading Ollama model for AI briefings "
                "(first launch may take a few minutes)…"
            ):
                preload_ollama_model(client)
        else:
            preload_ollama_model(client)
        status = "ready"
    except Exception as exc:
        status = f"failed: {exc}"
    finally:
        st.session_state[_DONE_KEY] = True
        st.session_state[_STATUS_KEY] = status

    return status


def get_ollama_client():
    """Return OllamaClient with a fresh module reload (avoids stale Streamlit session classes)."""
    ensure_ollama_preloaded(show_spinner=False)
    return create_ollama_client()


def render_ollama_sidebar_status() -> None:
    """Show LLM preload status in the sidebar."""
    if not st.session_state.get(_DONE_KEY):
        return
    status = str(st.session_state.get(_STATUS_KEY, ""))
    if status == "ready":
        st.sidebar.success("AI model ready")
    elif status == "unavailable":
        st.sidebar.warning("Ollama offline — instant briefings only")
    elif status == "disabled":
        st.sidebar.caption("Ollama preload disabled")
    elif status.startswith("failed"):
        st.sidebar.error("AI model preload failed")
    else:
        st.sidebar.info(f"AI model: {status}")
