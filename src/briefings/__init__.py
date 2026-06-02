"""LLM-powered maintenance briefing generation."""

from .briefing_prompts import (
    SYSTEM_PROMPT,
    build_alert_summary_prompt,
    build_briefing_prompt,
    build_instant_briefing,
)
from .ollama_client import OllamaClient

__all__ = [
    "OllamaClient",
    "SYSTEM_PROMPT",
    "build_briefing_prompt",
    "build_instant_briefing",
    "build_alert_summary_prompt",
]
