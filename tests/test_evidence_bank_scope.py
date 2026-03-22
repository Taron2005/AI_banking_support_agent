"""Bank-scoped evidence post-filter and used_sources alignment with LLM pool."""

from __future__ import annotations

from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk
from voice_ai_banking_support_agent.runtime.orchestrator import RuntimeOrchestrator, RuntimeRequest
from voice_ai_banking_support_agent.runtime.session_state import SessionStateStore


def _rc(bank_key: str, url: str, *, topic: str = "deposit") -> RetrievedChunk:
    return RetrievedChunk(
        score=0.85,
        chunk=DocumentMetadata(
            bank_key=bank_key,
            bank_name=bank_key,
            topic=topic,
            source_url=url,
            page_title="P",
            section_title="S",
            language="hy",
            chunk_id=f"{bank_key}-1",
            raw_text="r",
            cleaned_text=f"Ավանդի մասին {bank_key} տեքստ տոկոս 5%։",
        ),
    )


class LeakyMultiBankRetriever:
    """Ignores RetrievalRequest.bank_keys (simulates bad retriever); tests orchestrator post-filter."""

    def retrieve(self, req):  # noqa: ANN001
        return [
            _rc("acba", "https://acba.am/deposits/term"),
            _rc("ameriabank", "https://ameriabank.am/deposits/classic"),
            _rc("idbank", "https://idbank.am/savings/demand"),
        ]


def test_single_named_bank_post_filter_and_sources_exclude_competitors() -> None:
    orch = RuntimeOrchestrator(retriever=LeakyMultiBankRetriever())  # type: ignore[arg-type]
    store = SessionStateStore()
    state = store.get_or_create("scope-1")
    out = orch.handle(
        RuntimeRequest(
            session_id="scope-1",
            query="Ամերիաբանկում ինչ ավանդներ կան, տոկոսները ասա",
            index_name="i",
        ),
        state,
    )
    assert out.status == "answered"
    joined = " ".join(out.used_sources).lower()
    assert "ameriabank.am" in joined or "ameria" in joined
    assert "acba.am" not in joined
    assert "idbank" not in joined


def test_two_named_banks_drop_unmentioned_bank_from_sources() -> None:
    orch = RuntimeOrchestrator(retriever=LeakyMultiBankRetriever())  # type: ignore[arg-type]
    store = SessionStateStore()
    state = store.get_or_create("scope-2")
    out = orch.handle(
        RuntimeRequest(
            session_id="scope-2",
            query="Ameriabank և IDBank ավանդների տոկոսադրույքը համեմատի",
            index_name="i",
        ),
        state,
    )
    assert out.status == "answered"
    joined = " ".join(out.used_sources).lower()
    assert "acba.am" not in joined
    assert ("ameriabank.am" in joined or "idbank" in joined)
