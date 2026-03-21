from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.bank_detector import BankDetector
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk, SessionState
from voice_ai_banking_support_agent.runtime.orchestrator import (
    RuntimeOrchestrator,
    RuntimeRequest,
    _resolve_bank_keys,
)
from voice_ai_banking_support_agent.runtime.session_state import SessionStateStore


class FakeRetriever:
    def retrieve(self, req):  # noqa: ANN001
        if "best bank" in req.query.lower():
            return []
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


class CapturingRetriever(FakeRetriever):
    def __init__(self) -> None:
        self.last_req = None

    def retrieve(self, req):  # noqa: ANN001
        self.last_req = req
        return super().retrieve(req)


def test_no_bank_in_query_does_not_pin_session_bank_without_followup_merge() -> None:
    cap = CapturingRetriever()
    orch = RuntimeOrchestrator(retriever=cap)  # type: ignore[arg-type]
    store = SessionStateStore()
    state = store.get_or_create("c1")
    state.last_bank = "acba"
    out = orch.handle(
        RuntimeRequest(session_id="c1", query="Ինչ ավանդներ կան հիմա", index_name="i"),
        state,
    )
    assert out.status == "answered"
    assert cap.last_req is not None
    assert cap.last_req.bank_keys is None


def test_followup_merge_applies_bank_when_last_bank_merged() -> None:
    cap = CapturingRetriever()
    orch = RuntimeOrchestrator(retriever=cap)  # type: ignore[arg-type]
    store = SessionStateStore()
    state = store.get_or_create("c2")
    state.last_bank = "ameriabank"
    state.last_topic = "deposit"
    out = orch.handle(
        RuntimeRequest(session_id="c2", query="իսկ տոկոսադրույքը?", index_name="i"),
        state,
    )
    assert out.status == "answered"
    assert cap.last_req is not None
    assert cap.last_req.bank_keys == frozenset({"ameriabank"})


def test_two_named_banks_yield_allowlist_not_single_bank() -> None:
    cap = CapturingRetriever()
    orch = RuntimeOrchestrator(retriever=cap)  # type: ignore[arg-type]
    store = SessionStateStore()
    state = store.get_or_create("cmp")
    out = orch.handle(
        RuntimeRequest(
            session_id="cmp",
            query="ACBA և Ameriabank ավանդների մասին համեմատի",
            index_name="i",
        ),
        state,
    )
    assert out.status == "answered"
    assert cap.last_req is not None
    assert cap.last_req.bank_keys == frozenset({"acba", "ameriabank"})
    assert set(out.detected_banks) == {"acba", "ameriabank"}
    assert out.detected_bank is None


def test_resolve_bank_keys_two_named_before_broad_compare_phrase() -> None:
    det = BankDetector()
    q = "ACBA և Ameriabank համեմատի ավանդները"
    allm = det.detect_all(q)
    keys = _resolve_bank_keys(
        effective_query=q,
        detected_all=allm,
        followup_merged_fields=[],
        state=SessionState(session_id="t"),
    )
    assert keys == frozenset({"acba", "ameriabank"})


def test_resolve_bank_keys_compare_without_names_opens_scope() -> None:
    det = BankDetector()
    q = "համեմատի ավանդները բոլոր բանկերում"
    allm = det.detect_all(q)
    keys = _resolve_bank_keys(
        effective_query=q,
        detected_all=allm,
        followup_merged_fields=[],
        state=SessionState(session_id="t2"),
    )
    assert keys is None
