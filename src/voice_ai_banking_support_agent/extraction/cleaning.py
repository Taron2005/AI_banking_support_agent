from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Tuple

from bs4 import BeautifulSoup
from bs4.element import Tag

from ..scrapers.base import BankExtractionRules

logger = logging.getLogger(__name__)


NOISE_LINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"copyright", re.IGNORECASE),
    re.compile(r"privacy policy", re.IGNORECASE),
    re.compile(r"terms of use", re.IGNORECASE),
    re.compile(r"cookie", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    # Armenian/other common footer-ish markers
    re.compile(r"քաղաքացի", re.IGNORECASE),
]


@dataclass(frozen=True)
class CleaningResult:
    """Result of cleaning HTML to normalized text."""

    raw_text: str
    cleaned_text: str
    usable: bool
    warning: str | None = None


def _remove_decorative_elements(soup: BeautifulSoup, rules: BankExtractionRules | None = None) -> None:
    """Remove nav/footer/header scripts/styles and common noise blocks."""

    # Remove elements by tag.
    for tag in soup.find_all(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Remove elements by class/id heuristics.
    noise_markers = {"menu", "breadcrumb", "cookies", "cookie", "footer", "header", "social"}
    for el in soup.find_all(True):
        # BeautifulSoup tags may have `attrs=None` depending on parser/content.
        # Guard against that so we never crash the pipeline.
        attrs_dict = el.attrs or {}
        cls_val = attrs_dict.get("class", "")
        # BS4 sometimes stores class as list[str].
        if isinstance(cls_val, list):
            cls_str = " ".join(str(x) for x in cls_val)
        else:
            cls_str = str(cls_val or "")
        id_val = attrs_dict.get("id", "")
        id_str = str(id_val or "")
        attrs = f"{cls_str} {id_str}".lower()
        if any(m in attrs for m in noise_markers):
            el.decompose()

    # Apply bank-specific selector removals.
    if rules is not None:
        for rule in rules.remove_selectors:
            try:
                for el in soup.select(rule.selector):
                    el.decompose()
            except Exception:
                # Selector errors should never fail ingestion; continue best-effort.
                logger.debug("Invalid remove selector ignored: %s", rule.selector)


def _extract_from_preferred_containers(
    soup: BeautifulSoup, rules: BankExtractionRules | None = None
) -> str | None:
    """
    Try extracting text from bank-preferred content containers.

    Returns None when no preferred container matches, so caller can fallback
    to full-page extraction.
    """

    if rules is None or not rules.prefer_content_selectors:
        return None

    parts: list[str] = []
    for rule in rules.prefer_content_selectors:
        try:
            for el in soup.select(rule.selector):
                if not isinstance(el, Tag):
                    continue
                txt = el.get_text(separator="\n", strip=True)
                if txt:
                    parts.append(txt)
        except Exception:
            logger.debug("Invalid preferred selector ignored: %s", rule.selector)

    if not parts:
        return None
    return normalize_text("\n".join(parts))


def _split_lines(text: str) -> list[str]:
    # Normalize newlines then split into lines to filter noise.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [ln.strip() for ln in text.split("\n")]


def normalize_text(text: str) -> str:
    """Normalize extracted text to improve retrieval consistency."""

    # Collapse whitespace everywhere.
    text = text.replace("\u00A0", " ")  # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip space around newlines.
    text = re.sub(r" *\n *", "\n", text)
    # Normalize repeated punctuation (very common in scraped pages).
    text = re.sub(r"([.!?…])\1{1,}", r"\1", text)
    return text.strip()


def remove_noise_lines(lines: Iterable[str]) -> Tuple[list[str], list[str]]:
    """
    Remove lines that are likely boilerplate or too short/noisy.

    Returns:
        (kept_lines, removed_reasons)
    """

    kept: list[str] = []
    removed_reasons: list[str] = []
    for ln in lines:
        if not ln:
            continue
        lower = ln.lower()

        if any(p.search(ln) for p in NOISE_LINE_PATTERNS):
            removed_reasons.append(f"noise_pattern:{lower[:30]}")
            continue

        # Drop extremely short lines unless they contain digits (often part of addresses/hours).
        if len(ln) < 8 and not re.search(r"\d", ln):
            removed_reasons.append(f"too_short:{lower[:30]}")
            continue

        # Drop lines that are mostly punctuation.
        if re.fullmatch(r"[\W_]+", ln):
            removed_reasons.append(f"punct_only:{lower[:30]}")
            continue

        kept.append(ln)
    return kept, removed_reasons


def is_text_useful(text: str) -> tuple[bool, str | None]:
    """Heuristic validation for whether extraction produced usable content."""

    t = normalize_text(text)
    if len(t) < 400:
        return False, "too_short"

    # If it has almost no letters/digits, it's likely a failure.
    letter_count = sum(1 for ch in t if ch.isalpha())
    digit_count = sum(1 for ch in t if ch.isdigit())
    if letter_count < 50 and digit_count < 10:
        return False, "low_information"

    return True, None


def detect_language_from_text(text: str) -> str:
    """
    Very lightweight script-based language hint.

    Returns:
        "hy" if Armenian script characters are dominant enough,
        otherwise "non_hy".
    """

    if not text.strip():
        return "non_hy"
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "non_hy"
    armenian_count = sum(1 for ch in letters if "\u0530" <= ch <= "\u058F")
    ratio = armenian_count / max(len(letters), 1)
    return "hy" if ratio >= 0.25 else "non_hy"


def clean_html_to_text(html: str, rules: BankExtractionRules | None = None) -> CleaningResult:
    """
    Extract and clean main page text from HTML.

    Args:
        html: Raw HTML string.

    Returns:
        CleaningResult with `raw_text` (extracted before normalization),
        and `cleaned_text` (final normalized for downstream extraction/chunking).
    """

    soup = BeautifulSoup(html, "lxml")
    _remove_decorative_elements(soup, rules=rules)

    # Prefer bank-specific main containers when present.
    preferred_text = _extract_from_preferred_containers(soup, rules=rules)

    # Use newline separator to preserve some structure.
    raw_text = preferred_text if preferred_text is not None else soup.get_text(separator="\n")
    raw_text = normalize_text(raw_text)

    lines = _split_lines(raw_text)
    kept_lines, _removed_reasons = remove_noise_lines(lines)
    cleaned_text = normalize_text("\n".join(kept_lines))

    usable, warning = is_text_useful(cleaned_text)
    if not usable:
        logger.warning("Extraction produced low-value content: warning=%s", warning)

    return CleaningResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        usable=usable,
        warning=warning,
    )

