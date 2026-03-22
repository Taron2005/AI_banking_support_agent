"""Bank clarification: short follow-up (bank name only) completes the prior question."""

from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk
from voice_ai_banking_support_agent.runtime.orchestrator import RuntimeOrchestrator, RuntimeRequest
from voice_ai_banking_support_agent.runtime.runtime_config import OrchestrationSettings
from voice_ai_banking_support_agent.runtime.session_state import SessionStateStore


class _CapRetriever:
    def __init__(self) -> None:
        self.last_query = ""
        self.last_bank_keys = None

    def retrieve(self, req):  # noqa: ANN001
        self.last_query = req.query
        self.last_bank_keys = req.bank_keys
        return [
            RetrievedChunk(
                score=0.9,
                chunk=DocumentMetadata(
                    bank_key="ameriabank",
                    bank_name="Ameriabank",
                    topic=req.topic or "deposit",
                    source_url="https://ameriabank.am/deposit",
                    page_title="Deposits",
                    section_title="Ameria deposit",
                    language="hy",
                    chunk_id="x1",
                    raw_text="raw",
                    cleaned_text="Ավանդի պայմաններ, տոկոսադրույք, Հասցե՝ Երևան",
                ),
            )
        ]


def test_bank_name_after_clarify_merges_and_pins_bank() -> None:
    cap = _CapRetriever()
    orch = RuntimeOrchestrator(
        retriever=cap,  # type: ignore[arg-type]
        orchestration=OrchestrationSettings(require_explicit_bank=True),
        bank_aliases={"acba": ["acba"], "ameriabank": ["ameria", "ameriabank"], "idbank": ["idbank"]},
    )
    st = SessionStateStore().get_or_create("pc1")
    r1 = orch.handle(
        RuntimeRequest(session_id="pc1", query="Ինչ ավանդներ կան հիմա", index_name="i"),
        st,
    )
    assert r1.status == "clarify"
    assert st.pending_clarify == "bank"
    assert st.pending_query
    assert st.pending_topic == "deposit"

    r2 = orch.handle(RuntimeRequest(session_id="pc1", query="Ամերիա", index_name="i"), st)
    assert r2.status == "answered"
    assert "ameriabank" in cap.last_query.lower() or "ամերիա" in cap.last_query.lower()
    assert cap.last_bank_keys == frozenset({"ameriabank"})
    assert st.pending_clarify is None


def test_all_banks_phrase_after_clarify_clears_pending() -> None:
    cap = _CapRetriever()
    orch = RuntimeOrchestrator(
        retriever=cap,  # type: ignore[arg-type]
        orchestration=OrchestrationSettings(require_explicit_bank=True),
        bank_aliases={"acba": ["acba"], "ameriabank": ["ameria"], "idbank": ["idbank"]},
    )
    st = SessionStateStore().get_or_create("pc2")
    orch.handle(RuntimeRequest(session_id="pc2", query="Ինչ վարկեր կան", index_name="i"), st)
    assert st.pending_clarify == "bank"

    r2 = orch.handle(
        RuntimeRequest(session_id="pc2", query="բոլոր բանկերում ասա", index_name="i"),
        st,
    )
    assert r2.status == "answered"
    assert cap.last_bank_keys is None
    assert st.pending_clarify is None
