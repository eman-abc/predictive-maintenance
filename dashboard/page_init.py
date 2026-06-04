"""Shared startup for every Streamlit page."""

from __future__ import annotations

import dashboard.bootstrap  # noqa: F401

import streamlit as st

from dashboard.ollama_startup import ensure_ollama_preloaded, render_ollama_sidebar_status
from dashboard.theme import apply_streamlit_theme


def init_page(*, show_legend: bool = True) -> None:
    """Run once-per-session theme, Ollama preload, and sidebar status."""
    apply_streamlit_theme()
    if show_legend:
        from dashboard.theme import render_semantic_legend

        render_semantic_legend()
    first_load = not st.session_state.get("ollama_preload_done", False)
    ensure_ollama_preloaded(show_spinner=first_load)
    render_ollama_sidebar_status()
