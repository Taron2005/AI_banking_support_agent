from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..runtime.models import RuntimeResponse

AudioEncoding = Literal["pcm_s16le", "wav", "mp3", "opus", "text"]


@dataclass(frozen=True)
class STTInput:
    """Input payload for STT providers (audio or pre-transcribed text)."""

    content: bytes
    encoding: AudioEncoding = "pcm_s16le"
    language: str = "hy-AM"


@dataclass(frozen=True)
class TTSOutput:
    """Output payload produced by TTS provider."""

    audio: bytes
    encoding: AudioEncoding


@dataclass(frozen=True)
class VoiceTurnResult:
    """Result of one voice turn after runtime orchestration."""

    session_id: str
    user_text: str
    runtime_response: RuntimeResponse
    tts_output: TTSOutput

