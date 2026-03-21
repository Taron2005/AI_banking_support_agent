from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from ..runtime.factory import build_runtime_orchestrator
from ..runtime.llm_config import LLMSettings
from ..runtime.runtime_config import RuntimeSettings
from .stt import HTTPWhisperSTTProvider, MockSTTProvider, STTProvider
from .tts import HTTPTTSProvider, MockTTSProvider, TTSProvider
from .voice_config import VoiceConfig


@dataclass(frozen=True)
class VoiceDependencies:
    stt: STTProvider
    tts: TTSProvider


def build_voice_dependencies(voice_config: VoiceConfig) -> VoiceDependencies:
    # Providers are config-swappable; mock remains deterministic default.
    mock_stt = MockSTTProvider()
    mock_tts = MockTTSProvider(encoding=voice_config.tts.output_encoding)

    if voice_config.stt.provider == "http_whisper":
        stt: STTProvider = HTTPWhisperSTTProvider(
            endpoint=voice_config.stt.endpoint or "",
            language=voice_config.stt.language,
            timeout_seconds=voice_config.stt.timeout_seconds,
            api_key=voice_config.stt.api_key,
            api_key_header=voice_config.stt.api_key_header,
            response_text_field=voice_config.stt.response_text_field,
            fallback_provider=mock_stt if voice_config.stt.fallback_to_mock else None,
        )
    else:
        stt = mock_stt

    if voice_config.tts.provider == "http_tts":
        tts: TTSProvider = HTTPTTSProvider(
            endpoint=voice_config.tts.endpoint or "",
            language=voice_config.tts.language,
            voice_name=voice_config.tts.voice_name,
            output_encoding=voice_config.tts.output_encoding,
            timeout_seconds=voice_config.tts.timeout_seconds,
            api_key=voice_config.tts.api_key,
            api_key_header=voice_config.tts.api_key_header,
            response_audio_field=voice_config.tts.response_audio_field,
            fallback_provider=mock_tts if voice_config.tts.fallback_to_mock else None,
        )
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

