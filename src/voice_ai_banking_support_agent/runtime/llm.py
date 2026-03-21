from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import requests

from .llm_config import LLMSettings

logger = logging.getLogger(__name__)

# Groq uses OpenAI-compatible /v1/chat/completions; system message enforces evidence-only answers.
LLM_SYSTEM_ARMENIAN_EVIDENCE_ONLY = (
    "Դու բանկային աջակցության օգնական ես։ Պատասխանիր միայն «Ապացույցներ» բլոկի տեքստից։ "
    "Մի օգտագործիր արտաքին գիտելիք կամ ընդհանուր գիտելիք բանկերի մասին։ "
    "Մի կրկնիր աղբյուրի URL-ները կամ ցուցակների չափազանց մեծ մեջբերումները։ "
    "Գրիր հայերեն, բնական և հակիրճ (2–6 կարճ նախադասություն), միայն հարցին վերաբերող փաստերով։ "
    "Եթե օգտվողի հարցը հայերեն է՝ պահպանիր նույն լեզվի ոճը, առանց թարգմանելու ապացույցի մեջբերումները։ "
    "Եթե ապացույցը բավարար չէ՝ մեկ նախադասությամբ նշիր դա, առանց գուշակելու։"
)

GROQ_DEFAULT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


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
class GroqChatClient:
    """Groq chat via OpenAI-compatible HTTP API."""

    endpoint: str
    api_key: str | None
    model: str
    timeout_seconds: float
    temperature: float
    max_tokens: int = 512
    system_message: str = LLM_SYSTEM_ARMENIAN_EVIDENCE_ONLY

    def generate(self, prompt: str) -> str:
        if not self.endpoint:
            raise RuntimeError("Groq endpoint is not configured.")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
            raise RuntimeError("Groq response content is empty.")
        return out.strip()


def build_llm_client(settings: LLMSettings | None) -> LLMClient | None:
    if settings is None:
        return None
    if settings.provider == "mock":
        return MockLLMClient()
    if settings.provider == "groq":
        import os

        key = settings.api_key or os.getenv("GROQ_API_KEY")
        if not key:
            logger.warning("Groq provider selected but no api_key or GROQ_API_KEY; LLM calls will fail.")
        return GroqChatClient(
            endpoint=settings.endpoint or GROQ_DEFAULT_ENDPOINT,
            api_key=key,
            model=settings.model,
            timeout_seconds=settings.timeout_seconds,
            temperature=settings.temperature,
        )
    logger.warning("Unknown LLM provider %s; expected mock or groq.", getattr(settings, "provider", None))
    return None
