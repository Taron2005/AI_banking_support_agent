from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import AppConfig
from ..runtime.factory import build_runtime_orchestrator
from ..runtime.llm_config import LLMSettings
from ..runtime.runtime_config import RuntimeSettings
from .stt import HTTPWhisperSTTProvider, MockSTTProvider, STTProvider
from .tts import HTTPTTSProvider, MockTTSProvider, TTSProvider
from .voice_config import VoiceConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceDependencies:
    stt: STTProvider
    tts: TTSProvider


def build_voice_dependencies(voice_config: VoiceConfig) -> VoiceDependencies:
    mock_stt = MockSTTProvider()
    mock_tts = MockTTSProvider(encoding=voice_config.tts.output_encoding)

    stt_ep = (voice_config.stt.endpoint or "").strip()
    if voice_config.stt.provider == "http_whisper" and stt_ep:
        stt: STTProvider = HTTPWhisperSTTProvider(
            endpoint=stt_ep,
            language=voice_config.stt.language,
            timeout_seconds=voice_config.stt.timeout_seconds,
            api_key=voice_config.stt.api_key,
            api_key_header=voice_config.stt.api_key_header,
            response_text_field=voice_config.stt.response_text_field,
            multipart_field=voice_config.stt.multipart_field,
            upload_filename=voice_config.stt.upload_filename,
            fallback_provider=mock_stt if voice_config.stt.fallback_to_mock else None,
        )
    elif voice_config.stt.provider == "http_whisper":
        logger.warning(
            "Voice STT: http_whisper selected but VOICE_STT_ENDPOINT is empty — using mock STT. "
            "Set VOICE_STT_ENDPOINT for real Armenian speech-to-text."
        )
        stt = mock_stt
    else:
        stt = mock_stt

    tts_ep = (voice_config.tts.endpoint or "").strip()
    if voice_config.tts.provider == "http_tts" and tts_ep:
        tts: TTSProvider = HTTPTTSProvider(
            endpoint=tts_ep,
            language=voice_config.tts.language,
            voice_name=voice_config.tts.voice_name,
            output_encoding=voice_config.tts.output_encoding,
            timeout_seconds=voice_config.tts.timeout_seconds,
            api_key=voice_config.tts.api_key,
            api_key_header=voice_config.tts.api_key_header,
            response_audio_field=voice_config.tts.response_audio_field,
            fallback_provider=mock_tts if voice_config.tts.fallback_to_mock else None,
        )
    elif voice_config.tts.provider == "http_tts":
        logger.warning(
            "Voice TTS: http_tts selected but VOICE_TTS_ENDPOINT is empty — using mock TTS (silence). "
            "Set VOICE_TTS_ENDPOINT for real Armenian text-to-speech."
        )
        tts = mock_tts
    else:
        tts = mock_tts

    return VoiceDependencies(stt=stt, tts=tts)


def build_runtime_for_voice(
    *,
    app_config: AppConfig,
    runtime_settings: RuntimeSettings,
    llm_settings: LLMSettings | None = None,
):
    return build_runtime_orchestrator(
        app_config=app_config, runtime_settings=runtime_settings, llm_settings=llm_settings
    )
