from __future__ import annotations

import base64
import io
import logging
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


@dataclass
class MockTTSProvider:
    """
    Mock TTS provider for local development/testing.

    Returns a short valid WAV (silence) so LiveKit/audio players do not break.
    """

    encoding: str = "wav"

    def synthesize(self, text: str) -> TTSOutput:
        _ = text  # length could scale duration in future; keep short for latency
        return TTSOutput(audio=_silent_wav_bytes(), encoding=self.encoding)


@dataclass
class HTTPTTSProvider:
    """
    HTTP TTS adapter for self-hosted TTS services.

    Supports:
    - direct audio bytes response
    - JSON response containing base64 audio field (default `audio_base64`)
    """

    endpoint: str
    language: str
    voice_name: str
    output_encoding: str = "wav"
    timeout_seconds: float = 20.0
    api_key: str | None = None
    api_key_header: str = "Authorization"
    response_audio_field: str = "audio_base64"
    fallback_provider: TTSProvider | None = None

    def synthesize(self, text: str) -> TTSOutput:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.api_key_header] = self.api_key
        payload = {
            "text": text,
            "language": self.language,
            "voice": self.voice_name,
            "format": self.output_encoding,
        }
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            ctype = (resp.headers.get("content-type") or "").lower()
            if "application/json" in ctype:
                body = resp.json()
                b64 = body.get(self.response_audio_field)
                if not isinstance(b64, str) or not b64.strip():
                    raise RuntimeError("TTS JSON response missing audio field")
                audio = base64.b64decode(b64)
            else:
                audio = resp.content
            if not audio:
                raise RuntimeError("TTS returned empty audio payload")
            return TTSOutput(audio=audio, encoding=self.output_encoding)  # type: ignore[arg-type]
        except Exception as exc:
            logger.exception("HTTP TTS failed: %s", exc)
            if self.fallback_provider is not None:
                return self.fallback_provider.synthesize(text)
            raise RuntimeError("TTS synthesis failed and no fallback is configured.") from exc

