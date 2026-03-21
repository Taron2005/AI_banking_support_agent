from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LiveKitSettings(BaseModel):
    url: str = ""
    api_key: str = "devkey"
    api_secret: str = "secret"
    room_name: str = "banking-support-room"
    agent_identity: str = "banking-support-agent"
    participant_identity_prefix: str = "user"


class STTSettings(BaseModel):
    # Default path: HTTP Whisper-style service; set VOICE_STT_ENDPOINT in .env.
    provider: Literal["mock", "http_whisper"] = "http_whisper"
    language: str = "hy"
    endpoint: str | None = None
    timeout_seconds: float = 180.0
    api_key: str | None = None
    api_key_header: str = "Authorization"
    response_text_field: str = "text"
    multipart_field: str = "file"
    upload_filename: str = "audio.wav"
    fallback_to_mock: bool = True


class TTSSettings(BaseModel):
    # Default path: HTTP TTS; set VOICE_TTS_ENDPOINT in .env.
    provider: Literal["mock", "http_tts"] = "http_tts"
    language: str = "hy-AM"
    voice_name: str = "default"
    output_encoding: Literal["wav", "mp3", "pcm_s16le"] = "wav"
    endpoint: str | None = None
    timeout_seconds: float = 60.0
    api_key: str | None = None
    api_key_header: str = "Authorization"
    response_audio_field: str = "audio_base64"
    fallback_to_mock: bool = True


class VoiceBehaviorSettings(BaseModel):
    debug: bool = False
    verbose_trace: bool = False
    max_response_chars: int = 4000


class VoiceConfig(BaseModel):
    livekit: LiveKitSettings = Field(default_factory=LiveKitSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    behavior: VoiceBehaviorSettings = Field(default_factory=VoiceBehaviorSettings)


def _coerce_livekit_ws_url(url: str) -> str:
    """RTC SDKs expect ws:// or wss://; .env often mistakenly uses http://."""

    u = (url or "").strip().rstrip("/")
    lower = u.lower()
    if lower.startswith("http://"):
        return "ws://" + u[7:]
    if lower.startswith("https://"):
        return "wss://" + u[8:]
    return u


def _validate_self_hosted_url(url: str) -> None:
    lower = url.strip().lower()
    if not lower:
        raise ValueError(
            "LIVEKIT_URL is required for self-hosted LiveKit usage. "
            "Set it via voice config or environment."
        )
    if not (lower.startswith("ws://") or lower.startswith("wss://") or lower.startswith("http://") or lower.startswith("https://")):
        raise ValueError("LIVEKIT_URL must start with ws://, wss://, http://, or https://.")
    cloud_markers = ("livekit.cloud", "cloud.livekit.io")
    if any(marker in lower for marker in cloud_markers):
        raise ValueError(
            "LiveKit Cloud URL detected. This project is self-hosted/open-source LiveKit only."
        )


def load_voice_config(path: str | Path | None = None) -> VoiceConfig:
    raw: dict = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = VoiceConfig.model_validate(raw)
    env_url = os.getenv("LIVEKIT_URL")
    env_key = os.getenv("LIVEKIT_API_KEY")
    env_secret = os.getenv("LIVEKIT_API_SECRET")
    stt_endpoint = (os.getenv("VOICE_STT_ENDPOINT") or "").strip() or None
    stt_key = (os.getenv("VOICE_STT_API_KEY") or "").strip() or None
    tts_endpoint = (os.getenv("VOICE_TTS_ENDPOINT") or "").strip() or None
    tts_key = (os.getenv("VOICE_TTS_API_KEY") or "").strip() or None
    if env_url:
        cfg.livekit.url = env_url
    cfg.livekit.url = _coerce_livekit_ws_url(cfg.livekit.url)
    if env_key:
        cfg.livekit.api_key = env_key
    if env_secret:
        cfg.livekit.api_secret = env_secret
    if stt_endpoint:
        cfg.stt.endpoint = stt_endpoint
    if stt_key:
        cfg.stt.api_key = stt_key
    stt_to = (os.getenv("VOICE_STT_TIMEOUT_SECONDS") or "").strip()
    if stt_to:
        try:
            cfg.stt.timeout_seconds = float(stt_to)
        except ValueError:
            pass
    if tts_endpoint:
        cfg.tts.endpoint = tts_endpoint
    if tts_key:
        cfg.tts.api_key = tts_key
    tts_to = (os.getenv("VOICE_TTS_TIMEOUT_SECONDS") or "").strip()
    if tts_to:
        try:
            cfg.tts.timeout_seconds = float(tts_to)
        except ValueError:
            pass
    _validate_self_hosted_url(cfg.livekit.url)
    force_mock = (os.getenv("VOICE_USE_MOCK") or "").strip().lower() in ("1", "true", "yes")
    if force_mock:
        cfg.stt.provider = "mock"
        cfg.tts.provider = "mock"
    return cfg

