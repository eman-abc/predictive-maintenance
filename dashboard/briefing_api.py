"""Stable briefing imports for Streamlit (avoids package __init__ cache issues)."""

from __future__ import annotations

import importlib

import dashboard.bootstrap  # noqa: F401 — must run before src imports

from src.briefings.briefing_prompts import (
    SYSTEM_PROMPT,
    build_briefing_prompt,
    build_instant_briefing,
)


def create_ollama_client():
    """Build OllamaClient from a freshly loaded module (avoids stale .pyc)."""
    import src.briefings.ollama_client as ollama_mod

    importlib.reload(ollama_mod)
    return ollama_mod.OllamaClient()


def preload_ollama_model(client) -> None:
    """Load the LLM into memory."""
    warmup = getattr(client, "warmup", None)
    if callable(warmup):
        warmup()
        return
    client.generate("ok")


__all__ = [
    "create_ollama_client",
    "SYSTEM_PROMPT",
    "build_briefing_prompt",
    "build_instant_briefing",
    "preload_ollama_model",
]
