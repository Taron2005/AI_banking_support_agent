from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.evidence_select import (
    dedupe_urls,
    normalize_http_url,
    rerank_and_select,
)
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk


def test_normalize_http_url_trailing_slash() -> None:
    a = normalize_http_url("https://Example.AM/foo/bar/")
    b = normalize_http_url("https://example.am/foo/bar")
    assert a == b


def test_dedupe_urls_order() -> None:
    out = dedupe_urls(
        [
            "https://a.am/x",
            "https://a.am/x/",
            "https://b.am/y",
        ],
        max_n=5,
    )
    assert len(out) == 2


def _mk_chunk(
    score: float,
    text: str,
    *,
    bank_key: str = "acba",
    bank_name: str = "ACBA",
    chunk_id: str = "c",
    source_url: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        score=score,
        chunk=DocumentMetadata(
            bank_key=bank_key,
            bank_name=bank_name,
            topic="deposit",
            source_url=source_url or f"https://{bank_key}.am",
            page_title="p",
            section_title="s",
            language="hy",
            chunk_id=chunk_id,
            raw_text=text,
            cleaned_text=text,
        ),
    )


def test_rerank_diversify_keeps_multiple_banks() -> None:
    body = (
        "Ավանդի պայմանները ներառում են տոկոսադրույք և ժամկետային սահմանափակումներ բանկի կայքում նշված ձևով։ "
    )
    acba = _mk_chunk(0.95, body + " ACBA", chunk_id="c1")
    ameri = _mk_chunk(0.72, body + " Ameriabank", bank_key="ameriabank", bank_name="Ameriabank", chunk_id="c2")
    idb = _mk_chunk(0.70, body + " IDBank", bank_key="idbank", bank_name="IDBank", chunk_id="c3")
    out = rerank_and_select([acba, ameri, idb], "ավանդների տոկոսադրույք", top_k=3, diversify_banks=True)
    keys = {c.chunk.bank_key for c in out}
    assert len(keys) >= 2


def test_rerank_diversify_respects_per_source_url_cap() -> None:
    body = (
        "Ավանդի պայմանները ներառում են տոկոսադրույք և ժամկետային սահմանափակումներ բանկի կայքում նշված ձևով։ "
    )
    same_url = "https://acba.am/deposits/classic"
    acba_a = _mk_chunk(0.95, body + " a", chunk_id="c1", source_url=same_url)
    acba_b = _mk_chunk(0.94, body + " b", chunk_id="c2", source_url=same_url)
    acba_c = _mk_chunk(0.93, body + " c", chunk_id="c3", source_url="https://acba.am/deposits/other")
    ameri = _mk_chunk(0.70, body + " Ameriabank", bank_key="ameriabank", bank_name="Ameriabank", chunk_id="c4")
    out = rerank_and_select(
        [acba_a, acba_b, acba_c, ameri],
        "ավանդների տոկոսադրույք",
        top_k=4,
        diversify_banks=True,
        max_per_bank=3,
        max_per_source_url=1,
    )
    urls = [c.chunk.source_url for c in out]
    assert urls.count(same_url) <= 1


def test_rerank_demotes_nav_like_chunk() -> None:
    nav = _mk_chunk(
        0.62,
        "home menu search login twitter facebook cookie privacy terms " * 3,
    )
    good = _mk_chunk(
        0.58,
        "Ավանդի տոկոսադրույքը սահմանվում է պայմանագրով և կարող է հասնել մինչև 8 տոկոսի։",
    )
    out = rerank_and_select([nav, good], "տոկոսադրույք ավանդ", top_k=1)
    assert "տոկոսադրույք" in out[0].chunk.cleaned_text
