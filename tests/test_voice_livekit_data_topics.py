"""Contract tests for LiveKit data-channel topics (voice UI + agent)."""

from voice_ai_banking_support_agent.voice.livekit_agent import (
    TOPIC_ASSISTANT_TEXT,
    TOPIC_ASSISTANT_TEXT_DELTA,
    TOPIC_PTT,
    TOPIC_VOICE_STATE,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
)


def test_voice_data_topics_stable() -> None:
    assert TOPIC_PTT == "voice.ptt"
    assert TOPIC_VOICE_STATE == "voice.state"
    assert TOPIC_VOICE_TRANSCRIPT_FINAL == "voice.transcript.final"
    assert TOPIC_ASSISTANT_TEXT == "assistant.text"
    assert TOPIC_ASSISTANT_TEXT_DELTA == "assistant.text.delta"
