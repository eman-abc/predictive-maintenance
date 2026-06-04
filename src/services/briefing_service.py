"""Briefing and metric explanation generation (Ollama in API layer only)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from src.briefings.briefing_prompts import (
    SYSTEM_PROMPT as BRIEFING_SYSTEM,
    build_briefing_prompt,
    build_instant_briefing,
)
from src.briefings.metric_explanation_prompts import (
    SECTION_SUMMARIZE,
    SYSTEM_PROMPT as METRIC_SYSTEM,
    build_explanation_prompt,
    build_instant_explanation,
    build_section_payload,
    build_summarizer_payload,
    build_summarizer_prompt,
)


def _ollama_client():
    import importlib
    import src.briefings.ollama_client as ollama_mod

    importlib.reload(ollama_mod)
    return ollama_mod.OllamaClient()


def ollama_status() -> dict[str, Any]:
    client = _ollama_client()
    return {
        "available": client.is_available(),
        "model": client.model,
        "base_url": client.base_url,
    }


def preload_ollama() -> dict[str, str]:
    client = _ollama_client()
    if not client.is_available():
        return {"status": "unavailable"}
    try:
        warmup = getattr(client, "warmup", None)
        if callable(warmup):
            warmup()
        else:
            client.generate("ok")
        return {"status": "ready"}
    except Exception as exc:
        return {"status": f"failed: {exc}"}


def _instant_from_context(context: dict[str, Any]) -> str:
    return build_instant_briefing(
        asset_id=str(context["asset_id"]),
        health_score=float(context["health_score"]),
        rul=float(context["rul"]),
        failure_probability=float(context["failure_probability"]),
        alert_level=str(context.get("alert_level", "normal")),
        recommended_action=str(context.get("recommended_action", "")),
        anomaly_score=float(context.get("anomaly_score", 0)),
        rul_pred_cox=context.get("rul_pred_cox"),
        survival_prob_30=context.get("survival_prob_30"),
    )


def _prompt_from_context(context: dict[str, Any]) -> str:
    return build_briefing_prompt(
        str(context["asset_id"]),
        float(context["health_score"]),
        float(context["rul"]),
        float(context["failure_probability"]),
        sensor_summary=context.get("sensor_summary"),
        rul_pred_cox=context.get("rul_pred_cox"),
        survival_prob_30=context.get("survival_prob_30"),
    )


def generate_briefing(
    *,
    mode: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    if mode == "instant":
        text = _instant_from_context(context)
        return {"text": text, "source": "instant", "mode": mode}

    client = _ollama_client()
    if not client.is_available():
        text = _instant_from_context(context)
        return {"text": text, "source": "instant_fallback", "mode": mode}

    prompt = _prompt_from_context(context)
    try:
        text = client.generate(prompt, system=BRIEFING_SYSTEM)
        return {"text": text.strip(), "source": "ai", "mode": mode}
    except httpx.TimeoutException:
        text = _instant_from_context(context)
        return {"text": text, "source": "instant_timeout", "mode": mode}
    except Exception as exc:
        text = _instant_from_context(context)
        return {
            "text": text,
            "source": "instant_error",
            "mode": mode,
            "error": str(exc),
        }


def generate_metric_explanation(
    *,
    section_id: str,
    dataset_id: str,
    summary: dict[str, Any],
    registry_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if section_id == SECTION_SUMMARIZE:
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT_METRICS_SUMMARY", "450"))
    else:
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT_METRICS", "320"))

    client = _ollama_client()
    if not client.is_available():
        text = build_instant_explanation(section_id, summary, dataset_id)
        return {"text": text, "source": "instant", "section_id": section_id}

    if section_id == SECTION_SUMMARIZE:
        payload = build_summarizer_payload(summary, dataset_id, registry_entry=registry_entry)
        prompt = build_summarizer_prompt(payload)
    else:
        payload = build_section_payload(
            section_id, summary, dataset_id, registry_entry=registry_entry
        )
        prompt = build_explanation_prompt(section_id, payload)

    try:
        text = client.generate(prompt, system=METRIC_SYSTEM, num_predict=num_predict)
        return {"text": text.strip(), "source": "ai", "section_id": section_id}
    except Exception as exc:
        text = build_instant_explanation(section_id, summary, dataset_id)
        return {
            "text": text,
            "source": "instant_error",
            "section_id": section_id,
            "error": str(exc),
        }
