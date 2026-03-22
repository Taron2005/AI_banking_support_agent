import pytest

from voice_ai_banking_support_agent.voice.tts_speech_prepare import prepare_text_for_tts


def test_prepare_strips_http_and_sources_section() -> None:
    raw = (
        "Պատասխանի մարմինը այստեղ։\n"
        "Աղբյուրներ՝\n"
        "https://ameriabank.am/deposit\n"
    )
    out = prepare_text_for_tts(raw)
    assert "https://" not in out
    assert "Աղբյուրներ" not in out
    assert "Պատասխանի" in out


def test_prepare_keeps_embedded_english_words() -> None:
    raw = "ACBA բանկում deposit տոկոսը 5% է։ Տեքստը https://x.com/y է։"
    out = prepare_text_for_tts(raw)
    assert "ACBA" in out
    assert "deposit" in out
    assert "https://" not in out


def test_prepare_markdown_link_keeps_label() -> None:
    raw = "Տես [ACBA վարկ](https://acba.am/loan)։"
    out = prepare_text_for_tts(raw)
    assert "ACBA" in out
    assert "https://" not in out


@pytest.mark.parametrize(
    ("src", "bad"),
    [
        ("Դիտեք https://example.com/path", "example.com"),
        ("կայքը www.test.am է", "www.test.am"),
    ],
)
def test_prepare_removes_inline_urls(src: str, bad: str) -> None:
    out = prepare_text_for_tts(src)
    assert bad not in out


def test_prepare_replaces_markdown_table() -> None:
    raw = "| Անվանում | Տոկոս |\n|----------|--------|\n| 12 ամիս | 9,5 |\nՀաջորդ նախադասությունը։"
    out = prepare_text_for_tts(raw)
    assert "Աղյուսակ" in out
    assert "|" not in out
    assert "Հաջորդ" in out


def test_prepare_rewrites_decimal_comma_and_dot() -> None:
    assert "ամբողջ" in prepare_text_for_tts("տոկոսը 22,5 տոկոս է։")
    assert "ամբողջ" in prepare_text_for_tts("տոկոսը 26,78 է։")
    assert "ամբողջ" in prepare_text_for_tts("rate 12.5 percent")


def test_prepare_expands_common_abbreviations() -> None:
    out = prepare_text_for_tts("Գումարը 5 մլն դրամ է ՀՀ դրամով։")
    assert "միլիոն" in out
    assert "Հայաստանի Հանրապետություն" in out
