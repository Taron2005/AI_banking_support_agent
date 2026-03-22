from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Protocol

import requests

from .voice_models import STTInput

logger = logging.getLogger(__name__)

# Returned for binary audio when mock STT is used (no endpoint or HTTP fallback).
MOCK_STT_PLACEHOLDER = "[mock-stt-unavailable]"


def is_mock_stt_placeholder(text: str | None) -> bool:
    return (text or "").strip() == MOCK_STT_PLACEHOLDER


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
        raw_audio = bytes(payload.content)
        data = {"language": lang}
        # (connect_timeout, read_timeout): local Whisper often needs >45s on CPU / cold model load.
        read_to = float(self.timeout_seconds)
        connect_to = min(20.0, max(5.0, read_to * 0.15))
        timeout = (connect_to, read_to)
        delays = (0.6, 2.0, 5.0, 8.0)
        last_exc: Exception | None = None
        for attempt in range(len(delays) + 1):
            # New buffer each attempt — requests consumes the stream; retries must not send empty bodies.
            audio_buf = io.BytesIO(raw_audio)
            files = {
                self.multipart_field: (
                    self.upload_filename,
                    audio_buf,
                    "audio/wav",
                ),
            }
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=timeout,
                )
                resp.raise_for_status()
                if not resp.encoding:
                    resp.encoding = resp.apparent_encoding or "utf-8"
                ctype = (resp.headers.get("content-type") or "").lower()
                if "application/json" not in ctype:
                    text = resp.content.decode("utf-8", errors="replace").strip()
                    if text:
                        return text
                    raise RuntimeError("STT returned non-JSON empty body")
                body = resp.json()
                text = _extract_transcript(body, self.response_text_field)
                if not text:
                    # Server may return 200 with empty text while still busy; retry once like gateway timeouts.
                    raise RuntimeError("STT returned empty transcript")
                return text
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt < len(delays):
                    time.sleep(delays[attempt])
                    continue
            except requests.HTTPError as exc:
                last_exc = exc
                code = exc.response.status_code if exc.response is not None else None
                if code in (502, 503, 504, 524) and attempt < len(delays):
                    time.sleep(delays[attempt])
                    continue
                break
            except RuntimeError as exc:
                last_exc = exc
                if "empty transcript" in str(exc).lower() and attempt < len(delays):
                    time.sleep(delays[attempt])
                    continue
                break
            except Exception as exc:
                last_exc = exc
                break
        exc: BaseException = last_exc if last_exc is not None else RuntimeError("STT request failed")
        hint = ""
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            try:
                hint = (exc.response.text or "")[:300]
            except Exception:
                hint = ""
        logger.warning(
            "HTTP STT request failed (%s). Endpoint=%s%s — using fallback if configured.",
            exc,
            self.endpoint,
            f" body={hint!r}" if hint else "",
        )
        logger.debug("STT failure detail", exc_info=True)
        if self.fallback_provider is not None:
            logger.info("STT: falling back to secondary provider.")
            return self.fallback_provider.transcribe(payload)
        raise RuntimeError(
            "STT transcription failed. Check VOICE_STT_ENDPOINT, local voice_http_stt_server logs, "
            "and mic audio (speak clearly; avoid empty buffers after PTT)."
        ) from exc
