"""Shared startup for every Streamlit page."""

from __future__ import annotations

import dashboard.bootstrap  # noqa: F401

import streamlit as st

from dashboard.ollama_startup import ensure_ollama_preloaded, render_ollama_sidebar_status


def init_page() -> None:
    """Run once-per-session Ollama preload and show sidebar status."""
    first_load = not st.session_state.get("ollama_preload_done", False)
    ensure_ollama_preloaded(show_spinner=first_load)
    render_ollama_sidebar_status()
