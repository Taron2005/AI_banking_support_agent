"""Optional orchestration flags (evaluation-style)."""

from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk
from voice_ai_banking_support_agent.runtime.orchestrator import RuntimeOrchestrator, RuntimeRequest
from voice_ai_banking_support_agent.runtime.runtime_config import OrchestrationSettings
from voice_ai_banking_support_agent.runtime.session_state import SessionStateStore


def _chunk(bank_key: str, topic: str = "deposit") -> RetrievedChunk:
    return RetrievedChunk(
        score=0.9,
        chunk=DocumentMetadata(
            bank_key=bank_key,
            bank_name=bank_key,
            topic=topic,
            source_url=f"https://{bank_key}.example/p",
            page_title="P",
            section_title="S",
            language="hy",
            chunk_id=f"{bank_key}-1",
            raw_text="r",
            cleaned_text="Ավանդի մասին տեքստ։",
        ),
    )


class TwoBankRetriever:
    def retrieve(self, req):  # noqa: ANN001
        return [_chunk("acba"), _chunk("idbank")]


class OneBankRetriever:
    def retrieve(self, req):  # noqa: ANN001
        return [_chunk("acba")]


def test_clarify_when_unscoped_multi_bank_evidence() -> None:
    orch = RuntimeOrchestrator(
        retriever=TwoBankRetriever(),  # type: ignore[arg-type]
        orchestration=OrchestrationSettings(clarify_when_unscoped_multi_bank_evidence=True),
    )
    store = SessionStateStore()
    out = orch.handle(
        RuntimeRequest(session_id="x", query="Ինչ ավանդներ կան հիմա", index_name="i"),
        store.get_or_create("x"),
    )
    assert out.status == "clarify"
    assert "բանկ" in out.answer_text.lower()


def test_refuse_comparison_without_two_banks_in_evidence() -> None:
    orch = RuntimeOrchestrator(
        retriever=OneBankRetriever(),  # type: ignore[arg-type]
        orchestration=OrchestrationSettings(refuse_comparison_without_multi_bank_evidence=True),
    )
    store = SessionStateStore()
    out = orch.handle(
        RuntimeRequest(
            session_id="y",
            query="Ո՞ր բանկն ունի ավելի բարձր տոկոսադրույք ավանդի համար",
            index_name="i",
        ),
        store.get_or_create("y"),
    )
    assert out.status == "refused"
    assert out.refusal_reason == "comparison_insufficient"
