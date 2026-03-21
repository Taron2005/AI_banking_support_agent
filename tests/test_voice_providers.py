from voice_ai_banking_support_agent.voice.stt import MockSTTProvider
from voice_ai_banking_support_agent.voice.tts import MockTTSProvider
from voice_ai_banking_support_agent.voice.voice_models import STTInput


def test_mock_stt_text_roundtrip() -> None:
    stt = MockSTTProvider()
    text = stt.transcribe(STTInput(content="Բարև".encode("utf-8"), encoding="text"))
    assert "Բարև" in text


def test_mock_tts_returns_bytes() -> None:
    tts = MockTTSProvider()
    out = tts.synthesize("text")
    assert isinstance(out.audio, bytes)
    assert len(out.audio) > 0


def test_mock_stt_non_text_fallback() -> None:
    stt = MockSTTProvider(fallback_text="[x]")
    text = stt.transcribe(STTInput(content=b"\x00\x01", encoding="wav"))
    assert text == "[x]"

