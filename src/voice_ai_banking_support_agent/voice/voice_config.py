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
    timeout_seconds: float = 120.0
    api_key: str | None = None
    api_key_header: str = "Authorization"
    response_text_field: str = "text"
    multipart_field: str = "file"
    upload_filename: str = "audio.wav"
    # When an HTTP endpoint is set, factory.py does not use mock STT fallback.
    fallback_to_mock: bool = False


class TTSSettings(BaseModel):
    # Default path: HTTP TTS; set VOICE_TTS_ENDPOINT in .env.
    provider: Literal["mock", "http_tts"] = "http_tts"
    language: str = "hy-AM"
    voice_name: str = "default"
    output_encoding: Literal["wav", "mp3", "pcm_s16le"] = "wav"
    # When output_encoding is pcm_s16le, raw bytes are interpreted at this rate (Edge/local stack uses 24 kHz).
    pcm_s16le_sample_rate: int = 24000
    endpoint: str | None = None
    timeout_seconds: float = 60.0
    api_key: str | None = None
    api_key_header: str = "Authorization"
    response_audio_field: str = "audio_base64"
    fallback_to_mock: bool = True


class VoiceBehaviorSettings(BaseModel):
    debug: bool = False
    verbose_trace: bool = False
    max_response_chars: int = 16000
    # Must match ``AudioSource`` on the published assistant track (WebRTC / LiveKit expect a fixed rate).
    livekit_publish_sample_rate: int = 24000
    # When True, sleep between LiveKit frames (~real-time). False = push audio faster (lower latency).
    livekit_playout_realtime_pacing: bool = False
    livekit_playout_frame_ms: float = 10.0
    # Same as text chat POST /chat top_k (keep in sync with App.jsx when you change retrieval depth).
    chat_top_k: int = 8
    # Gemini token streaming over LiveKit data channel + TTS on final scrubbed answer (in-process only).
    stream_llm_tokens: bool = False
    # When true, voice turns call FastAPI POST /chat (same session_id as the web UI) — shared SessionState.
    route_through_runtime_api: bool = True
    runtime_api_url: str = "http://127.0.0.1:8000"
    runtime_api_timeout_seconds: float = 180.0
    # After PTT end: wait for remote mic track if browser publishes audio slightly after "start".
    mic_track_wait_seconds: float = 0.7
    # After PTT end: keep recording while waiting so trailing WebRTC audio is still buffered.
    # Should be >= browser unpublish delay (~220ms in App.jsx) minus network jitter; default 0.30s.
    pcm_trail_pause_seconds: float = 0.30


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
    tts_pcm_sr = (os.getenv("VOICE_TTS_PCM_SAMPLE_RATE") or "").strip()
    if tts_pcm_sr:
        try:
            cfg.tts.pcm_s16le_sample_rate = int(tts_pcm_sr)
        except ValueError:
            pass
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
    http_rt = (os.getenv("VOICE_RUNTIME_HTTP") or "").strip().lower()
    if http_rt in ("0", "false", "no", "off"):
        cfg.behavior.route_through_runtime_api = False
    api_u = (os.getenv("VOICE_RUNTIME_API_URL") or "").strip()
    if api_u:
        cfg.behavior.runtime_api_url = api_u.rstrip("/")
    api_to = (os.getenv("VOICE_RUNTIME_API_TIMEOUT_SECONDS") or "").strip()
    if api_to:
        try:
            cfg.behavior.runtime_api_timeout_seconds = float(api_to)
        except ValueError:
            pass
    mic_w = (os.getenv("VOICE_MIC_TRACK_WAIT_SECONDS") or "").strip()
    if mic_w:
        try:
            cfg.behavior.mic_track_wait_seconds = float(mic_w)
        except ValueError:
            pass
    trail = (os.getenv("VOICE_PCM_TRAIL_PAUSE_SECONDS") or "").strip()
    if trail:
        try:
            cfg.behavior.pcm_trail_pause_seconds = float(trail)
        except ValueError:
            pass
    pub_sr = (os.getenv("VOICE_LIVEKIT_PUBLISH_SAMPLE_RATE") or "").strip()
    if pub_sr:
        try:
            cfg.behavior.livekit_publish_sample_rate = int(pub_sr)
        except ValueError:
            pass
    pace = (os.getenv("VOICE_LIVEKIT_PLAYOUT_PACING") or "").strip().lower()
    if pace in ("0", "false", "no", "off"):
        cfg.behavior.livekit_playout_realtime_pacing = False
    elif pace in ("1", "true", "yes", "on"):
        cfg.behavior.livekit_playout_realtime_pacing = True
    chat_k = (os.getenv("VOICE_CHAT_TOP_K") or "").strip()
    if chat_k:
        try:
            cfg.behavior.chat_top_k = max(1, min(int(chat_k), 64))
        except ValueError:
            pass
    return cfg

