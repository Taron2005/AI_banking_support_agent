from __future__ import annotations

import re
import unicodedata

# Conservative repairs for local Whisper STT on Armenian + bank names (reduces spurious clarify / ambiguous).
_STT_SUBSTRING_FIXES: tuple[tuple[str, str], ...] = (
    ("վարք", "վարկ"),
    ("վարքեր", "վարկեր"),
    ("աբանդ", "ավանդ"),
    ("ամերիբանկ", "ամերիաբանկ"),
    ("ամերիյաբանկ", "ամերիաբանկ"),
    ("ամերիա բանկ", "ամերիաբանկ"),
    ("ամերիաբանկ", "ամերիաբանկ"),
    ("ակբա բանկ", "ակբա"),
    ("ակբաֆբանկ", "ակբա"),
    ("այդի բանկ", "իդբանկ"),
    ("այդիբանկ", "իդբանկ"),
    ("իդ բանկ", "իդբանկ"),
    ("իդբանկ", "իդբանկ"),
)


def normalize_query(text: str) -> str:
    text = (text or "").strip()
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text


def repair_stt_transcript(text: str) -> str:
    """Light post-processing after STT before topic/bank classification."""
    t = normalize_query(text)
    if not t:
        return t
    for wrong, right in _STT_SUBSTRING_FIXES:
        if re.search(re.escape(wrong), t, re.IGNORECASE):
            t = re.sub(re.escape(wrong), right, t, flags=re.IGNORECASE)
    return t

