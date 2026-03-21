from __future__ import annotations

import re
from typing import Iterable

from ..config import ChunkingConfig
from ..models import DocumentMetadata, TopicLabel
from ..extraction.section_parser import Section
from ..utils.text import slugify, stable_id


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into approximate sentences.

    Notes:
    - This is language-agnostic and intentionally simple.
    - It supports Armenian full stop `։` as a boundary.
    """

    text = text.replace("\n", " ")
    # Keep Armenian '։' and regular punctuation.
    # This is naive but works well for chunk sizing in many scraped pages.
    parts = re.split(r"(?<=[.!?\u0589])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def chunk_sections(
    *,
    sections: list[Section],
    bank_key: str,
    bank_name: str,
    topic: TopicLabel,
    source_url: str,
    page_title: str,
    language: str,
    raw_page_text: str,
    chunking: ChunkingConfig,
) -> list[DocumentMetadata]:
    """
    Create section-aware retrieval chunks.

    Each chunk is produced from exactly one section (never mixes unrelated sections).
    The section title is prepended into the chunk text for better retrieval grounding.
    """

    url_slug = slugify(source_url)
    docs: list[DocumentMetadata] = []

    for section_idx, section in enumerate(sections):
        section_start_doc_count = len(docs)
        sentences = _split_into_sentences(section.content_text)
        if not sentences:
            continue

        current_sentences: list[str] = []
        current_words = 0
        chunk_idx = 0

        def flush() -> None:
            nonlocal current_sentences, current_words, chunk_idx
            if not current_sentences:
                return
            chunk_body = " ".join(current_sentences).strip()
            chunk_words = _count_words(chunk_body)
            if chunk_words < chunking.min_words:
                # Short page/card sections can still be highly informative
                # (e.g. compact product cards, branch cards, rates lists).
                # Keep at least one short chunk per section when no prior chunk exists.
                if not (chunk_words >= 20 and len(docs) == section_start_doc_count):
                    current_sentences = []
                    current_words = 0
                    return

            heading_prefixed = f"{section.title}\n{chunk_body}".strip()
            chunk_id = stable_id(bank_name, topic, url_slug, section_idx, chunk_idx, heading_prefixed)
            docs.append(
                DocumentMetadata(
                    bank_key=bank_key,
                    bank_name=bank_name,
                    topic=topic,
                    source_url=source_url,
                    page_title=page_title,
                    section_title=section.title,
                    language=language,
                    chunk_id=chunk_id,
                    raw_text=raw_page_text,
                    cleaned_text=heading_prefixed,
                )
            )
            chunk_idx += 1
            current_sentences = []
            current_words = 0

        for sent in sentences:
            sent_words = _count_words(sent)
            if current_words + sent_words > chunking.max_words and current_sentences:
                flush()

            # If a sentence itself is too long, hard-split by words to respect max_words.
            if sent_words > chunking.max_words:
                words = re.findall(r"\S+", sent)
                for i in range(0, len(words), chunking.target_words):
                    part = " ".join(words[i : i + chunking.target_words])
                    if _count_words(part) >= chunking.min_words:
                        current_sentences = [part]
                        current_words = _count_words(part)
                        flush()
                continue

            current_sentences.append(sent)
            current_words += sent_words

            if current_words >= chunking.target_words:
                flush()

        # Flush remaining content.
        if current_sentences:
            flush()

    return docs

