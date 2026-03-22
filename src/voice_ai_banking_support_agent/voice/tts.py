from __future__ import annotations

import base64
import io
import logging
import time
import wave
from dataclasses import dataclass
from typing import Protocol

import requests

from .voice_models import TTSOutput

logger = logging.getLogger(__name__)


def _silent_wav_bytes(*, duration_s: float = 0.35, sample_rate: int = 24000) -> bytes:
    """Minimal valid mono s16le WAV for mock TTS (short silence)."""
    n = int(sample_rate * duration_s)
    pcm = b"\x00\x00" * n
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class TTSProvider(Protocol):
    def synthesize(self, text: str) -> TTSOutput: ...


def _extract_base64_audio(body: dict, primary_field: str) -> bytes | None:
    for key in (primary_field, "audio_base64", "audio", "data", "payload"):
        v = body.get(key)
        if isinstance(v, str) and v.strip():
            try:
                return base64.b64decode(v)
            except Exception:
                continue
    nested = body.get("audioContent") or body.get("content")
    if isinstance(nested, str) and nested.strip():
        try:
            return base64.b64decode(nested)
        except Exception:
            return None
    return None


@dataclass
class MockTTSProvider:
    """Short silent WAV for tests when no TTS HTTP endpoint is available."""

    encoding: str = "wav"

    def synthesize(self, text: str) -> TTSOutput:
        _ = text
        return TTSOutput(audio=_silent_wav_bytes(), encoding=self.encoding)


@dataclass
class HTTPTTSProvider:
    """
    HTTP TTS for JSON (base64 audio) or raw audio (audio/wav, audio/mpeg) responses.

    Request JSON (UTF-8): text, language, voice, format — suitable for Armenian text.
    """

    endpoint: str
    language: str
    voice_name: str
    output_encoding: str = "wav"
    timeout_seconds: float = 60.0
    api_key: str | None = None
    api_key_header: str = "Authorization"
    response_audio_field: str = "audio_base64"
    fallback_provider: TTSProvider | None = None

    def synthesize(self, text: str) -> TTSOutput:
        stripped = (text or "").strip()
        if not stripped:
            return TTSOutput(audio=_silent_wav_bytes(duration_s=0.2), encoding=self.output_encoding)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            key = self.api_key.strip()
            h = self.api_key_header.strip()
            if h.lower() == "authorization" and not key.lower().startswith("bearer "):
                headers[h] = f"Bearer {key}"
            else:
                headers[h] = key
        # JSON nulls or non-strings break strict TTS servers (422). Coerce defensively.
        def _s(v: object, default: str) -> str:
            if v is None:
                return default
            return v if isinstance(v, str) else str(v)

        payload = {
            "text": _s(stripped, ""),
            "language": _s(self.language, "hy-AM"),
            "voice": _s(self.voice_name, "default"),
            "format": _s(self.output_encoding, "wav"),
        }
        read_to = float(self.timeout_seconds)
        connect_to = min(12.0, max(4.0, read_to * 0.12))
        timeout = (connect_to, read_to)
        delays = (0.5, 1.5, 4.0)
        last_exc: Exception | None = None
        for attempt in range(len(delays) + 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                ctype = (resp.headers.get("content-type") or "").lower()
                if "application/json" in ctype:
                    body = resp.json()
                    audio = _extract_base64_audio(body, self.response_audio_field)
                    if not audio:
                        raise RuntimeError("TTS JSON response missing decodable audio field")
                else:
                    audio = resp.content
                if not audio:
                    raise RuntimeError("TTS returned empty audio payload")
                if self.output_encoding == "wav" and len(audio) >= 4 and audio[:4] != b"RIFF":
                    logger.warning("TTS payload may not be WAV despite format=wav (first bytes not RIFF).")
                return TTSOutput(audio=audio, encoding=self.output_encoding)  # type: ignore[arg-type]
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt < len(delays):
                    time.sleep(delays[attempt])
                    continue
            except requests.HTTPError as exc:
                last_exc = exc
                code = exc.response.status_code if exc.response is not None else None
                if code in (408, 429, 500, 502, 503, 504, 524) and attempt < len(delays):
                    time.sleep(delays[attempt])
                    continue
                break
            except Exception as exc:
                last_exc = exc
                break

        exc = last_exc if last_exc is not None else RuntimeError("TTS request failed")
        logger.exception("HTTP TTS failed after retries: %s", exc)
        if self.fallback_provider is not None:
            return self.fallback_provider.synthesize(text)
        raise RuntimeError("TTS synthesis failed and no fallback is configured.") from exc
