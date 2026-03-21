from voice_ai_banking_support_agent.voice.stt import _extract_transcript, normalize_whisper_language


def test_normalize_whisper_language() -> None:
    assert normalize_whisper_language("hy-AM") == "hy"
    assert normalize_whisper_language("HY") == "hy"
    assert normalize_whisper_language("en-US") == "en"


def test_extract_transcript() -> None:
    assert _extract_transcript({"text": "Բարև"}, "text") == "Բարև"
    assert _extract_transcript({"transcription": "hello"}, "text") == "hello"
    assert _extract_transcript({}, "text") == ""
