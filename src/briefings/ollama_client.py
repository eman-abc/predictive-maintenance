"""Ollama / HuggingFace API wrapper for maintenance briefings."""

import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class OllamaClient:
    """Generate text via a local Ollama server or compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Send a completion request to Ollama."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request to Ollama."""
        payload = {"model": self.model, "messages": messages, "stream": False}

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False
