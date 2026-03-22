"""
Fuzzy lexical matching for noisy STT / typos (Armenian + Latin tokens).
Shared by topic classification; bank detection uses its own difflib path.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

_TOKEN_RE = re.compile(r"[\w\u0561-\u0587\u0531-\u0556]+", re.UNICODE)


def normalize_for_match(text: str) -> str:
    return unicodedata.normalize("NFC", (text or "").strip()).casefold()


def fuzzy_term_matches(text_lower: str, term: str, *, ratio: float = 0.82) -> bool:
    """
    True if ``term`` appears literally in ``text_lower`` (already casefolded) or
    closely matches a token (SequenceMatcher) — tolerates 1–2 char STT/typo drift.
    """

    t = (term or "").strip().casefold()
    if len(t) < 2:
        return False
    if t in text_lower:
        return True
    if " " in t:
        if t in text_lower:
            return True
        parts = [p for p in t.split() if len(p) >= 2]
        return len(parts) >= 2 and all(
            fuzzy_term_matches(text_lower, p, ratio=min(0.88, ratio + 0.04)) for p in parts
        )
    max_delta = max(2, len(t) // 4)
    for tok in _TOKEN_RE.findall(text_lower):
        if abs(len(tok) - len(t)) > max_delta:
            continue
        if len(tok) < 3 and len(t) >= 6:
            continue
        if difflib.SequenceMatcher(None, tok, t).ratio() >= ratio:
            return True
    return False
