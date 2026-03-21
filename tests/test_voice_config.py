from pathlib import Path

import pytest

from voice_ai_banking_support_agent.voice.voice_config import load_voice_config


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


def test_voice_config_requires_endpoints_for_http_providers(tmp_path: Path) -> None:
    p = tmp_path / "voice.yaml"
    p.write_text("livekit:\n  url: ws://127.0.0.1:7880\nstt:\n  provider: http_whisper\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_voice_config(p)

