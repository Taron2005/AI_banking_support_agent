from __future__ import annotations

import re
from ..config import ChunkingConfig
from ..models import DocumentMetadata, TopicLabel
from ..extraction.section_parser import Section
from ..utils.text import slugify, stable_id


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into approximate sentences.

    Notes:
    - Supports Armenian full stop `։` as a boundary.
    - Does not split on abbreviations; kept simple for scraped bank pages.
    """

    text = text.replace("\n", " ")
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
    Sentence overlap reduces cuts mid-thought between consecutive chunks.
    """

    url_slug = slugify(source_url)
    docs: list[DocumentMetadata] = []
    overlap_n = max(0, int(chunking.overlap_sentences))

    for section_idx, section in enumerate(sections):
        section_start_doc_count = len(docs)
        sentences = _split_into_sentences(section.content_text)
        if not sentences:
            continue

        current_sentences: list[str] = []
        current_words = 0
        chunk_idx = 0

        def flush(*, force: bool = False) -> None:
            nonlocal current_sentences, current_words, chunk_idx
            if not current_sentences:
                return
            chunk_body = " ".join(current_sentences).strip()
            chunk_words = _count_words(chunk_body)
            if chunk_words < chunking.min_words and not force:
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
            if overlap_n > 0 and len(current_sentences) > overlap_n:
                tail = current_sentences[-overlap_n:]
                current_sentences = tail
                current_words = sum(_count_words(s) for s in tail)
            else:
                current_sentences = []
                current_words = 0

        for sent in sentences:
            sent_words = _count_words(sent)
            if current_words + sent_words > chunking.max_words and current_sentences:
                flush(force=False)

            if sent_words > chunking.max_words:
                words = re.findall(r"\S+", sent)
                for i in range(0, len(words), chunking.target_words):
                    part = " ".join(words[i : i + chunking.target_words])
                    current_sentences = [part]
                    current_words = _count_words(part)
                    flush(force=True)
                continue

            current_sentences.append(sent)
            current_words += sent_words

            if current_words >= chunking.target_words:
                flush(force=False)

        if current_sentences:
            flush(force=True)

        # If section produced no chunks (all sentences below min_words), keep one combined chunk.
        if len(docs) == section_start_doc_count and sentences:
            chunk_body = " ".join(sentences).strip()
            heading_prefixed = f"{section.title}\n{chunk_body}".strip()
            chunk_id = stable_id(bank_name, topic, url_slug, section_idx, 0, heading_prefixed)
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

    return docs
