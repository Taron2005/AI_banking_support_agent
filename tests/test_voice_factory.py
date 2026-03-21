from voice_ai_banking_support_agent.voice.factory import build_voice_dependencies
from voice_ai_banking_support_agent.voice.voice_config import VoiceConfig


def test_voice_factory_builds_mock_dependencies() -> None:
    deps = build_voice_dependencies(VoiceConfig())
    assert deps.stt is not None
    assert deps.tts is not None


def test_voice_factory_builds_http_providers() -> None:
    cfg = VoiceConfig.model_validate(
        {
            "livekit": {"url": "ws://127.0.0.1:7880"},
            "stt": {"provider": "http_whisper", "endpoint": "http://127.0.0.1:9001/stt"},
            "tts": {"provider": "http_tts", "endpoint": "http://127.0.0.1:9002/tts"},
        }
    )
    deps = build_voice_dependencies(cfg)
    assert deps.stt.__class__.__name__ == "HTTPWhisperSTTProvider"
    assert deps.tts.__class__.__name__ == "HTTPTTSProvider"

