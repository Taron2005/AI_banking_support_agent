"""LiveKit agent token resolution (no real LiveKit server)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from voice_ai_banking_support_agent.voice.livekit_agent import LiveKitVoiceAgent
from voice_ai_banking_support_agent.voice.voice_config import VoiceConfig


def _minimal_voice_config() -> VoiceConfig:
    return VoiceConfig.model_validate(
        {
            "livekit": {
                "url": "ws://127.0.0.1:7880",
                "api_key": "k",
                "api_secret": "s",
                "room_name": "r1",
                "agent_identity": "agent-1",
            }
        }
    )


def test_resolve_agent_token_prefers_livekit_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_TOKEN", "preissued-jwt")
    cfg = _minimal_voice_config()
    agent = object.__new__(LiveKitVoiceAgent)
    agent._voice_config = cfg
    assert LiveKitVoiceAgent._resolve_agent_token(agent) == "preissued-jwt"


def test_resolve_agent_token_mints_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_TOKEN", raising=False)
    cfg = _minimal_voice_config()
    agent = object.__new__(LiveKitVoiceAgent)
    agent._voice_config = cfg
    with patch(
        "voice_ai_banking_support_agent.runtime.livekit_tokens.mint_participant_token",
        return_value="minted-jwt",
    ) as m:
        tok = LiveKitVoiceAgent._resolve_agent_token(agent)
    assert tok == "minted-jwt"
    m.assert_called_once_with(
        identity="agent-1",
        room="r1",
        api_key="k",
        api_secret="s",
    )
