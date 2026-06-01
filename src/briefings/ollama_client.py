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
        timeout: float | None = None,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.timeout = float(
            timeout if timeout is not None else os.getenv("OLLAMA_TIMEOUT", "300")
        )

    def _http_timeout(self) -> httpx.Timeout:
        """Separate connect vs read limits; read covers model cold-start on slow hardware."""
        return httpx.Timeout(connect=10.0, read=self.timeout, write=30.0, pool=10.0)

    def _generation_options(self) -> dict[str, Any]:
        return {
            "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "400")),
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.4")),
        }

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Send a completion request to Ollama."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
            "options": self._generation_options(),
        }
        if system:
            payload["system"] = system

        with httpx.Client(timeout=self._http_timeout()) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")

    def warmup(self) -> None:
        """Load the model into memory so the first user-facing call is faster."""
        payload = {
            "model": self.model,
            "prompt": "ok",
            "stream": False,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
            "options": {"num_predict": 1},
        }
        with httpx.Client(timeout=self._http_timeout()) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request to Ollama."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
            "options": self._generation_options(),
        }

        with httpx.Client(timeout=self._http_timeout()) as client:
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
