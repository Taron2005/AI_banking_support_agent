from voice_ai_banking_support_agent.voice.tts_chunking import split_for_sequential_tts


def test_split_armenian_sentences() -> None:
    text = "Առաջին նախադասությունը։ Երկրորդը նույնպես։ Երրորդ։"
    parts = split_for_sequential_tts(text, max_chunk_chars=200)
    assert len(parts) >= 2
    joined = " ".join(parts)
    assert "Առաջին" in joined
    assert "Երրորդ" in joined


def test_split_merges_tiny_tail() -> None:
    text = "Կարճ։ Ք"
    parts = split_for_sequential_tts(text, min_chunk_chars=4)
    assert len(parts) == 1


def test_split_long_segment_by_words() -> None:
    words = ["բառ"] * 80
    text = " ".join(words)
    parts = split_for_sequential_tts(text, max_chunk_chars=40)
    assert len(parts) >= 2
    assert all(len(p) <= 120 for p in parts)
