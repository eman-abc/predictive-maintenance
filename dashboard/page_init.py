"""Shared startup for every Streamlit page."""

from __future__ import annotations

import dashboard.bootstrap  # noqa: F401

import streamlit as st

from dashboard.api_client import api_base_url, get_health, use_api_backend
from dashboard.theme import apply_streamlit_theme


def init_page(*, show_legend: bool = True) -> None:
    """Run once-per-session theme, backend status, and sidebar."""
    apply_streamlit_theme()
    if show_legend:
        from dashboard.theme import render_semantic_legend

        render_semantic_legend()

    if use_api_backend():
        _render_api_sidebar()
        return

    from dashboard.ollama_startup import ensure_ollama_preloaded, render_ollama_sidebar_status

    first_load = not st.session_state.get("ollama_preload_done", False)
    ensure_ollama_preloaded(show_spinner=first_load)
    render_ollama_sidebar_status()


def _render_api_sidebar() -> None:
    st.sidebar.caption(f"API: `{api_base_url()}`")
    try:
        health = get_health()
        if health.get("ollama_available"):
            st.sidebar.success(f"API + Ollama ({health.get('ollama_model')})")
        else:
            st.sidebar.warning("API up — Ollama offline (instant briefings only)")
    except Exception as exc:
        st.sidebar.error(f"API unreachable: {exc}")
