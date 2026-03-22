"""
Prepare assistant answer text for text-to-speech only.

UI / data-channel payloads keep the full answer (including «Աղբյուրներ» and URLs).
Spoken output must not read raw links, markdown, or source lists aloud.
Embedded Latin/English words are preserved for hybrid Armenian+English sentences.

Tables are replaced with a short Armenian cue so TTS does not read grid cells.
European-style decimal commas (e.g. 22,5) are rewritten for clearer speech.
"""

from __future__ import annotations

import re

# Body lines that start a sources section (strip from here to end for TTS).
_SOURCE_SECTION_START = re.compile(
    r"(?im)^\s*(«?\s*Աղբյուրներ\s*»?\s*[:՝.]?\s*|Աղբյուրներ\s*[:՝]\s*|sources?\s*[:.]?\s*|references?\s*[:.]?\s*)",
)
# Standalone URL lines
_URL_LINE = re.compile(r"^\s*https?://\S+\s*$", re.MULTILINE | re.IGNORECASE)
# Inline URLs
_INLINE_URL = re.compile(r"https?://[^\s)\]»\"']+", re.IGNORECASE)
_WWW_HOST = re.compile(r"\bwww\.[^\s)\]»\"']+", re.IGNORECASE)
# Markdown links: [label](url) -> label (keep English/Armenian label)
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]+\)")
# Repeated punctuation that confuses TTS
_REPEAT_PUNCT = re.compile(r"([.!?։…])\1{2,}")

_HTML_TABLE = re.compile(r"(?is)<table\b[^>]*>.*?</table>")
_MD_TABLE_ROW = re.compile(r"^\s*\|[^|\n]*(\|[^|\n]*)+\|\s*$")

_TABLE_PLACEHOLDER_HY = (
    "Աղյուսակ կա մանրամասն տվյալներով, որը ձայնով չի կարդացվում։ "
    "Դիտեք տեքստային պատասխանը։"
)

# European decimal comma: 22,5 or 26,78 — not thousands (no more digits immediately after).
_DECIMAL_COMMA = re.compile(r"(?<![\d.])(\d{1,6}),(\d{1,4})(?![\d,])")
# Latin decimal point in numeric context (e.g. copied from English PDFs).
_DECIMAL_DOT = re.compile(r"(?<![\d])(\d{1,6})\.(\d{1,4})(?!\d)")

# Abbreviations common in Armenian banking copy (longer keys first).
_ABBREV_SPEAK: tuple[tuple[str, str], ...] = (
    ("մլրդ", "միլիարդ"),
    ("մլն", "միլիոն"),
    ("դր․", "դրամ"),
    ("դր.", "դրամ"),
    ("ՀՀ", "Հայաստանի Հանրապետություն"),
    ("հհ", "Հայաստանի Հանրապետություն"),
)

_L_BOUND = r"(?:^|(?<=[\s\(\[«\"'՛՝՜.,;:!?\u0589—\-․]))"
_R_BOUND = r"(?=(?:[\s\)\]»\"'.,;:!?\u0589%—\-․]|$))"


def _replace_markdown_tables(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _MD_TABLE_ROW.match(lines[i]):
            j = i
            while j < len(lines) and _MD_TABLE_ROW.match(lines[j]):
                j += 1
            if j - i >= 2:
                out.append(_TABLE_PLACEHOLDER_HY)
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _rewrite_decimal_commas(text: str) -> str:
    """22,5 -> 22 ամբողջ 5 (clearer for TTS than a bare comma)."""

    def _sub(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return f"{a} ամբողջ {b}"

    t = _DECIMAL_COMMA.sub(_sub, text)
    return _DECIMAL_DOT.sub(_sub, t)


def _expand_abbreviations(text: str) -> str:
    t = text
    for abbr, spoken in _ABBREV_SPEAK:
        pat = _L_BOUND + re.escape(abbr) + _R_BOUND
        t = re.sub(pat, spoken, t, flags=re.IGNORECASE)
    return t


def prepare_text_for_tts(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").strip()
    t = _HTML_TABLE.sub(f" {_TABLE_PLACEHOLDER_HY} ", t)
    t = _replace_markdown_tables(t)
    t = _rewrite_decimal_commas(t)
    t = _expand_abbreviations(t)
    t = _MD_LINK.sub(r"\1", t)
    m = _SOURCE_SECTION_START.search(t)
    if m:
        t = t[: m.start()].rstrip()
    t = _INLINE_URL.sub(" ", t)
    t = _WWW_HOST.sub(" ", t)
    t = _URL_LINE.sub("", t)
    t = _REPEAT_PUNCT.sub(r"\1\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    t = " ".join(lines) if lines else ""
    t = re.sub(r"\s+([.!?։])", r"\1", t)
    return t.strip()
