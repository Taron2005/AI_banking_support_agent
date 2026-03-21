from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.evidence_pack import (
    dedupe_retrieved_chunks,
    prepare_evidence_for_answer,
    strip_navigation_lines,
)
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk


def _doc(chunk_id: str, text: str, url: str = "https://b.am/x") -> RetrievedChunk:
    return RetrievedChunk(
        score=0.5,
        chunk=DocumentMetadata(
            bank_key="b",
            bank_name="B",
            topic="deposit",
            source_url=url,
            page_title="p",
            section_title="s",
            language="hy",
            chunk_id=chunk_id,
            raw_text="raw",
            cleaned_text=text,
        ),
    )


def test_dedupe_drops_identical_chunk_id() -> None:
    a = _doc("same", "Ավանդի տոկոսադրույքը 5% է։")
    b = _doc("same", "Ավանդի տոկոսադրույքը 5% է։")
    out = dedupe_retrieved_chunks([a, b], max_keep=5)
    assert len(out) == 1


def test_dedupe_drops_near_duplicate_body() -> None:
    t = "Սպառողական վարկի տոկոսադրույքը և ժամկետը նշված են պայմանագրում։"
    a = _doc("c1", t)
    b = _doc("c2", t + " ")
    out = dedupe_retrieved_chunks([a, b], max_keep=5)
    assert len(out) == 1


def test_strip_navigation_lines_drops_short_nav_tokens() -> None:
    raw = "Տոկոսադրույքը 5% է\n«\nԳլխավոր\nՄանրամասն տեղեկություն"
    out = strip_navigation_lines(raw)
    assert "5%" in out
    assert "Մանրամասն" in out
    assert "Գլխավոր" not in out


def test_prepare_evidence_respects_max_chunks() -> None:
    chunks = [_doc(f"id{i}", f"տեքստ {i} " * 8) for i in range(10)]
    out = prepare_evidence_for_answer(chunks, max_chunks=3)
    assert len(out) <= 3
