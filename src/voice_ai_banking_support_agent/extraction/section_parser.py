from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from bs4 import BeautifulSoup
from bs4.element import Tag

from .cleaning import normalize_text
from ..scrapers.base import BankExtractionRules

logger = logging.getLogger(__name__)

HEADING_TAGS = ["h1", "h2", "h3"]


@dataclass(frozen=True)
class Section:
    """A parsed section from HTML."""

    title: str
    level: int
    content_text: str


def _prune_soup_for_sections(soup: BeautifulSoup, rules: BankExtractionRules | None = None) -> None:
    """Prune navigation/footer-like content before section extraction."""

    for tag in soup.find_all(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
        tag.decompose()

    noise_markers = {
        "menu",
        "breadcrumb",
        "cookies",
        "cookie",
        "footer",
        "header",
        "social",
        "banner",
        "contentinfo",
    }
    for el in soup.find_all(True):
        attrs_dict = el.attrs or {}
        cls_val = attrs_dict.get("class", "")
        if isinstance(cls_val, list):
            cls_str = " ".join(str(x) for x in cls_val)
        else:
            cls_str = str(cls_val or "")
        id_val = attrs_dict.get("id", "")
        id_str = str(id_val or "")
        attrs = f"{cls_str} {id_str}".lower()
        if any(m in attrs for m in noise_markers):
            el.decompose()

    if rules is not None:
        for rule in rules.remove_selectors:
            try:
                for el in soup.select(rule.selector):
                    el.decompose()
            except Exception:
                logger.debug("Invalid remove selector ignored in section parser: %s", rule.selector)


def _is_heading_tag(tag_name: str | None) -> bool:
    return tag_name is not None and tag_name.lower() in HEADING_TAGS


def _collect_text_from_elements(elements: Iterable) -> str:
    parts: list[str] = []
    for el in elements:
        if getattr(el, "get_text", None) is None:
            continue
        txt = el.get_text(separator=" ", strip=True)
        if txt:
            parts.append(txt)
    return normalize_text("\n".join(parts))


def _fallback_sections_from_blocks(
    soup: BeautifulSoup,
    *,
    rules: BankExtractionRules | None,
    min_content_chars: int,
    max_sections: int,
) -> list[Section]:
    """Build sections from repeated block/card selectors when headings are weak."""

    if rules is None or not rules.fallback_block_selectors:
        return []

    sections: list[Section] = []
    seen_texts: set[str] = set()

    for rule in rules.fallback_block_selectors:
        try:
            nodes = soup.select(rule.selector)
        except Exception:
            continue

        for node in nodes:
            if not isinstance(node, Tag):
                continue

            text = normalize_text(node.get_text(separator=" ", strip=True))
            if len(text) < min_content_chars:
                continue
            if text in seen_texts:
                continue

            heading_el = node.find(["h1", "h2", "h3", "h4", "strong", "b"])
            if heading_el is not None:
                title = heading_el.get_text(separator=" ", strip=True) or "Untitled"
            else:
                # Fallback title: first 8 words gives context for retrieval/debug.
                words = text.split()
                title = " ".join(words[:8]) if words else "Untitled"

            sections.append(Section(title=title, level=2, content_text=text))
            seen_texts.add(text)
            if len(sections) >= max_sections:
                return sections

    return sections


def parse_sections_from_html(
    html: str,
    *,
    max_sections: int = 50,
    min_content_chars: int = 80,
    rules: BankExtractionRules | None = None,
) -> list[Section]:
    """
    Parse HTML into heading-based sections.

    This is section-aware chunking input: each returned section should correspond
    to a heading and its associated content until the next heading.
    """

    soup = BeautifulSoup(html, "lxml")
    _prune_soup_for_sections(soup, rules=rules)

    headings = [h for h in soup.find_all(HEADING_TAGS)]
    if not headings:
        # First fallback: bank-specific repeated content blocks/cards.
        block_sections = _fallback_sections_from_blocks(
            soup, rules=rules, min_content_chars=min_content_chars, max_sections=max_sections
        )
        if block_sections:
            return block_sections

        # Second fallback: one "Untitled" section containing overall visible text.
        text = normalize_text(soup.get_text(separator="\n"))
        if text and len(text) >= min_content_chars:
            return [Section(title="Untitled", level=1, content_text=text)]
        return []

    sections: list[Section] = []
    for i, h in enumerate(headings[:max_sections]):
        title = h.get_text(separator=" ", strip=True)
        if not title:
            continue

        level = int(re.sub(r"[^0-9]", "", h.name or "1") or 1)
        level = {"h1": 1, "h2": 2, "h3": 3}.get((h.name or "").lower(), 1)

        # Collect siblings until the next heading.
        content_parts: list[str] = []
        for sib in h.next_siblings:
            if getattr(sib, "name", None) and _is_heading_tag(getattr(sib, "name", None)):
                break

            # Collect meaningful elements only.
            if getattr(sib, "name", None) is None:
                continue
            if sib.name in ["p", "li", "div", "span", "table", "section", "article"]:
                txt = sib.get_text(separator=" ", strip=True)
                if txt:
                    content_parts.append(txt)

        section_text = normalize_text("\n".join(content_parts))
        if len(section_text) < min_content_chars:
            continue

        sections.append(Section(title=title, level=level, content_text=section_text))

    if not sections:
        block_sections = _fallback_sections_from_blocks(
            soup, rules=rules, min_content_chars=min_content_chars, max_sections=max_sections
        )
        if block_sections:
            return block_sections
        text = normalize_text(soup.get_text(separator="\n"))
        if text and len(text) >= min_content_chars:
            return [Section(title="Untitled", level=1, content_text=text)]
    return sections

