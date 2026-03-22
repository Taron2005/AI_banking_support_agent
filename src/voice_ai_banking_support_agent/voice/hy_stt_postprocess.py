"""
Light cleanup and typo nudges for Armenian STT output before RAG.

Whisper often mis-spells domain words; we only apply safe Unicode hygiene plus a small,
conservative replacement table (whole phrases / frequent confusions), not aggressive
spell-correction that could change user intent.
"""

from __future__ import annotations

import re
import unicodedata

# Order matters: longer phrases first.
_STT_PHRASE_FIXES: tuple[tuple[str, str], ...] = (
    # Spacing / hyphen variants banks use on sites
    ("Ի Դ Բանկ", "ԻԴԲանկ"),
    ("Ի Դ բանկ", "ԻԴԲանկ"),
    ("ԻԴ բանկ", "ԻԴԲանկ"),
    ("Ամերիա բանկ", "Ամերիաբանկ"),
    ("Ամերիաբանք", "Ամերիաբանկ"),
    ("ամերիա բանկ", "Ամերիաբանկ"),
    # Common Latin-letter leaks in hy ASR for the same names
    ("ID bank", "ԻԴԲանկ"),
    ("ID Bank", "ԻԴԲանկ"),
    ("Idbank", "ԻԴԲանկ"),
    ("ACBA", "ԱԿԲԱ"),
)

_ZW_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")


def normalize_stt_transcript_hy(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _ZW_RE.sub("", t)
    t = t.replace("\xa0", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)
    for wrong, right in _STT_PHRASE_FIXES:
        if wrong in t:
            t = t.replace(wrong, right)
    return t.strip()
