from voice_ai_banking_support_agent.voice.hy_stt_postprocess import normalize_stt_transcript_hy


def test_normalize_strips_zw_and_nfkc() -> None:
    t = normalize_stt_transcript_hy("a\u200bb\u200c")
    assert t == "ab"


def test_normalize_bank_phrase_fixes() -> None:
    assert "ԻԴԲանկ" in normalize_stt_transcript_hy("Ի Դ Բանկում")
    assert normalize_stt_transcript_hy("Ամերիաբանք") == "Ամերիաբանկ"


def test_normalize_collapses_space() -> None:
    assert normalize_stt_transcript_hy("hello  \t  world") == "hello world"
