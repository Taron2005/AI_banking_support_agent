from pathlib import Path

import pytest

from voice_ai_banking_support_agent.voice.voice_config import load_voice_config


@pytest.fixture(autouse=True)
def _clear_livekit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "VOICE_STT_ENDPOINT",
        "VOICE_TTS_ENDPOINT",
    ):
        monkeypatch.delenv(k, raising=False)


def test_voice_config_defaults() -> None:
    with pytest.raises(ValueError):
        load_voice_config(None)


def test_voice_config_yaml_override(tmp_path: Path) -> None:
    p = tmp_path / "voice.yaml"
    p.write_text(
        "livekit:\n  url: ws://127.0.0.1:7880\n  room_name: test-room\nbehavior:\n  verbose_trace: true\n",
        encoding="utf-8",
    )
    cfg = load_voice_config(p)
    assert cfg.livekit.room_name == "test-room"
    assert cfg.behavior.verbose_trace is True


def test_voice_config_rejects_cloud_url(tmp_path: Path) -> None:
    p = tmp_path / "voice.yaml"
    p.write_text("livekit:\n  url: wss://demo.livekit.cloud\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_voice_config(p)


def test_voice_config_allows_http_providers_without_endpoint(tmp_path: Path) -> None:
    p = tmp_path / "voice.yaml"
    p.write_text("livekit:\n  url: ws://127.0.0.1:7880\nstt:\n  provider: http_whisper\n", encoding="utf-8")
    cfg = load_voice_config(p)
    assert cfg.stt.provider == "http_whisper"
    assert not cfg.stt.endpoint


def test_voice_config_stt_timeout_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICE_STT_TIMEOUT_SECONDS", "240")
    p = tmp_path / "voice.yaml"
    p.write_text("livekit:\n  url: ws://127.0.0.1:7880\n", encoding="utf-8")
    cfg = load_voice_config(p)
    assert cfg.stt.timeout_seconds == 240.0


def test_voice_config_tts_timeout_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICE_TTS_TIMEOUT_SECONDS", "90")
    p = tmp_path / "voice.yaml"
    p.write_text("livekit:\n  url: ws://127.0.0.1:7880\n", encoding="utf-8")
    cfg = load_voice_config(p)
    assert cfg.tts.timeout_seconds == 90.0


def test_voice_config_force_mock_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICE_USE_MOCK", "1")
    p = tmp_path / "voice.yaml"
    p.write_text(
        "livekit:\n  url: ws://127.0.0.1:7880\nstt:\n  provider: http_whisper\n  endpoint: http://x/stt\n",
        encoding="utf-8",
    )
    cfg = load_voice_config(p)
    assert cfg.stt.provider == "mock"
    assert cfg.tts.provider == "mock"

