from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import requests

from .voice_models import STTInput

logger = logging.getLogger(__name__)


class STTProvider(Protocol):
    def transcribe(self, payload: STTInput) -> str: ...


@dataclass
class MockSTTProvider:
    """
    Mock STT provider for local development/testing.

    Behavior:
    - if payload encoding is `text`, decode bytes directly
    - otherwise returns a deterministic placeholder marker
    """

    fallback_text: str = "[mock-stt-unavailable]"

    def transcribe(self, payload: STTInput) -> str:
        if payload.encoding == "text":
            return payload.content.decode("utf-8", errors="ignore").strip()
        return self.fallback_text


@dataclass
class HTTPWhisperSTTProvider:
    """
    HTTP STT adapter for self-hosted whisper-compatible services.

    Expected response JSON includes a text field (default key: `text`).
    """

    endpoint: str
    language: str
    timeout_seconds: float = 20.0
    api_key: str | None = None
    api_key_header: str = "Authorization"
    response_text_field: str = "text"
    fallback_provider: STTProvider | None = None

    def transcribe(self, payload: STTInput) -> str:
        if payload.encoding == "text":
            return payload.content.decode("utf-8", errors="ignore").strip()

        headers: dict[str, str] = {}
        if self.api_key:
            headers[self.api_key_header] = self.api_key
        files = {
            "file": ("audio.bin", payload.content, "application/octet-stream"),
        }
        data = {"language": payload.language or self.language}
        try:
            resp = requests.post(
                self.endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            body = resp.json()
            text = str(body.get(self.response_text_field, "")).strip()
            if not text:
                raise RuntimeError("STT returned empty transcript")
            return text
        except Exception as exc:
            logger.exception("HTTP STT failed: %s", exc)
            if self.fallback_provider is not None:
                return self.fallback_provider.transcribe(payload)
            raise RuntimeError("STT transcription failed and no fallback is configured.") from exc

