"""Tests for Ollama client wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from src.briefings.ollama_client import OllamaClient
from src.briefings.briefing_prompts import (
    SYSTEM_PROMPT,
    build_briefing_prompt,
    build_instant_briefing,
)


def test_build_briefing_prompt_contains_asset_info():
    prompt = build_briefing_prompt("ENG-001", 75.0, 50.0, 0.15)
    assert "ENG-001" in prompt
    assert "75" in prompt


def test_build_briefing_prompt_with_sensors():
    prompt = build_briefing_prompt(
        "ENG-001", 75.0, 50.0, 0.15, {"sensor_1": 600.0, "sensor_2": 1.0}
    )
    assert "sensor_1" in prompt
    assert len(prompt) < 500


def test_build_briefing_prompt_caps_sensors():
    prompt = build_briefing_prompt(
        "ENG-001",
        75.0,
        50.0,
        0.15,
        {f"sensor_{i}": float(i) for i in range(10)},
        max_sensors=2,
    )
    assert "sensor_0" in prompt
    assert "sensor_9" not in prompt


def test_build_instant_briefing():
    text = build_instant_briefing(
        "ENG-001", 72.0, 25.0, 0.34, alert_level="warning"
    )
    assert "ENG-001" in text
    assert "25" in text
    assert "34%" in text or "0.34" in text.lower() or "34" in text


@patch("src.briefings.ollama_client.httpx.Client")
def test_generate_calls_ollama_api(mock_client_cls):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Maintenance briefing text."}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value = mock_client

    client = OllamaClient(base_url="http://localhost:11434", model="llama3.2")
    result = client.generate("Test prompt", system=SYSTEM_PROMPT)
    assert result == "Maintenance briefing text."
    mock_client.post.assert_called_once()
