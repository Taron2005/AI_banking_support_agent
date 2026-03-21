from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import requests

from .llm_config import LLMSettings

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class MockLLMClient:
    """Deterministic local mock client for LLM integration tests."""

    prefix: str = "[MOCK-LLM]"

    def generate(self, prompt: str) -> str:
        text = prompt.replace("\n", " ")
        return f"{self.prefix} {text[:180]}..."


@dataclass
class OpenAICompatibleHTTPClient:
    """
    OpenAI-compatible /v1/chat/completions HTTP client.

    Works with self-hosted gateways exposing OpenAI-compatible API shape.
    """

    endpoint: str
    api_key: str | None
    model: str
    timeout_seconds: float
    temperature: float

    def generate(self, prompt: str) -> str:
        if not self.endpoint:
            raise RuntimeError("LLM endpoint is not configured.")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Answer in Armenian using evidence only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        resp = requests.post(self.endpoint, headers=headers, json=body, timeout=self.timeout_seconds)
        resp.raise_for_status()
        payload = resp.json()
        out = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not isinstance(out, str) or not out.strip():
            raise RuntimeError("LLM response content is empty.")
        return out.strip()


@dataclass
class GeminiRESTClient:
    """
    Gemini REST API client.

    Uses Google Generative Language REST endpoint with API key auth.
    """

    model: str
    api_key: str
    timeout_seconds: float
    temperature: float
    endpoint: str | None = None

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Gemini API key is not configured.")
        url = self.endpoint or (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        resp = requests.post(
            url,
            params={"key": self.api_key},
            json=body,
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        payload = resp.json()
        out = (
            payload.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not isinstance(out, str) or not out.strip():
            raise RuntimeError("Gemini response content is empty.")
        return out.strip()


def build_llm_client(settings: LLMSettings | None) -> LLMClient | None:
    if settings is None:
        return None
    if settings.provider == "mock":
        return MockLLMClient()
    if settings.provider == "openai_compatible_http":
        return OpenAICompatibleHTTPClient(
            endpoint=settings.endpoint or "",
            api_key=settings.api_key,
            model=settings.model,
            timeout_seconds=settings.timeout_seconds,
            temperature=settings.temperature,
        )
    if settings.provider == "gemini_rest":
        return GeminiRESTClient(
            model=settings.model or "gemini-2.0-flash",
            api_key=settings.api_key or "",
            timeout_seconds=settings.timeout_seconds,
            temperature=settings.temperature,
            endpoint=settings.endpoint,
        )
    logger.warning("Unknown LLM provider: %s", settings.provider)
    return None

