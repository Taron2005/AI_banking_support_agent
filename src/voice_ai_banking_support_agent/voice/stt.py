from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import requests

from .voice_models import STTInput

logger = logging.getLogger(__name__)


class STTProvider(Protocol):
    def transcribe(self, payload: STTInput) -> str: ...


def normalize_whisper_language(language: str | None) -> str:
    """
    Map BCP-47 style tags (e.g. hy-AM) to ISO-639-1 codes expected by Whisper-style APIs.
    """

    l = (language or "hy").strip().lower()
    if l.startswith("hy"):
        return "hy"
    if l.startswith("en"):
        return "en"
    if l.startswith("ru"):
        return "ru"
    if "-" in l:
        return l.split("-", 1)[0]
    return l or "hy"


def _extract_transcript(body: object, primary_field: str) -> str:
    if not isinstance(body, dict):
        return ""
    keys = (
        primary_field,
        "text",
        "transcription",
        "Transcription",
        "result",
        "transcript",
    )
    for k in keys:
        v = body.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Nested OpenAI-style
    segs = body.get("segments")
    if isinstance(segs, list) and segs:
        parts: list[str] = []
        for s in segs:
            if isinstance(s, dict) and isinstance(s.get("text"), str):
                parts.append(s["text"].strip())
        if parts:
            return " ".join(parts).strip()
    return ""


@dataclass
class MockSTTProvider:
    """
    Mock STT for automated tests or when no HTTP endpoint is configured.

    - `text` encoding: passthrough UTF-8 (Armenian-safe).
    - binary audio: returns a fixed marker (not real speech recognition).
    """

    fallback_text: str = "[mock-stt-unavailable]"

    def transcribe(self, payload: STTInput) -> str:
        if payload.encoding == "text":
            return payload.content.decode("utf-8", errors="ignore").strip()
        return self.fallback_text


@dataclass
class HTTPWhisperSTTProvider:
    """
    HTTP STT for Whisper-compatible or multipart /transcribe style services.

    Contract:
    - POST multipart form: default field `file` with audio bytes (WAV recommended).
    - Form field `language` set to a Whisper language code (hy, en, …).
    - JSON response: transcript in `text` or common alternates (see _extract_transcript).

    Responses must be UTF-8 JSON so Armenian is preserved.
    """

    endpoint: str
    language: str
    timeout_seconds: float = 45.0
    api_key: str | None = None
    api_key_header: str = "Authorization"
    response_text_field: str = "text"
    multipart_field: str = "file"
    upload_filename: str = "audio.wav"
    fallback_provider: STTProvider | None = None

    def transcribe(self, payload: STTInput) -> str:
        if payload.encoding == "text":
            return payload.content.decode("utf-8", errors="ignore").strip()

        headers: dict[str, str] = {}
        if self.api_key:
            key = self.api_key.strip()
            h = self.api_key_header.strip()
            if h.lower() == "authorization" and not key.lower().startswith("bearer "):
                headers[h] = f"Bearer {key}"
            else:
                headers[h] = key

        lang = normalize_whisper_language(payload.language or self.language)
        files = {
            self.multipart_field: (
                self.upload_filename,
                payload.content,
                "audio/wav",
            ),
        }
        data = {"language": lang}
        try:
            resp = requests.post(
                self.endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            ctype = (resp.headers.get("content-type") or "").lower()
            if "application/json" not in ctype:
                text = resp.content.decode("utf-8", errors="replace").strip()
                if text:
                    return text
                raise RuntimeError("STT returned non-JSON empty body")
            body = resp.json()
            text = _extract_transcript(body, self.response_text_field)
            if not text:
                raise RuntimeError("STT returned empty transcript")
            return text
        except Exception as exc:
            logger.warning(
                "HTTP STT request failed (%s). Endpoint=%s — using fallback if configured.",
                exc,
                self.endpoint,
            )
            logger.debug("STT failure detail", exc_info=True)
            if self.fallback_provider is not None:
                logger.info("STT: falling back to secondary provider.")
                return self.fallback_provider.transcribe(payload)
            raise RuntimeError(
                "STT transcription failed and no fallback is configured. "
                "Set VOICE_STT_ENDPOINT or enable fallback_to_mock in voice config."
            ) from exc
