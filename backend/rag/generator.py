"""
DocuMind-AI — LLM Generator

Unified interface for generating answers via Groq API or local Ollama.
Supports both streaming and non-streaming responses.
"""

import json
from typing import Generator

import httpx
import requests

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class LLMGenerator:
    """Generates text from an LLM (Groq API or local Ollama)."""

    def __init__(self):
        self.provider = settings.llm_provider.lower()
        logger.info(f"LLM generator initialised — provider: {self.provider}")

    # ------------------------------------------------------------------ Public
    def generate(self, messages: list[dict]) -> str:
        """Send messages to the LLM and return the full response text."""
        if self.provider == "groq":
            return self._generate_groq(messages)
        elif self.provider == "ollama":
            return self._generate_ollama(messages)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """Stream response tokens from the LLM."""
        if self.provider == "groq":
            yield from self._stream_groq(messages)
        elif self.provider == "ollama":
            yield from self._stream_ollama(messages)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    # ------------------------------------------------------------------ Groq
    def _generate_groq(self, messages: list[dict]) -> str:
        """Non-streaming Groq API call."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.groq_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _stream_groq(self, messages: list[dict]) -> Generator[str, None, None]:
        """Streaming Groq API call (SSE)."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.groq_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
            "stream": True,
        }

        with httpx.Client(timeout=60) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    # ------------------------------------------------------------------ Ollama
    def _generate_ollama(self, messages: list[dict]) -> str:
        """Non-streaming Ollama call."""
        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    def _stream_ollama(self, messages: list[dict]) -> Generator[str, None, None]:
        """Streaming Ollama call (NDJSON)."""
        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": True,
        }

        with httpx.Client(timeout=120) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
