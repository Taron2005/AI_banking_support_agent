from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk
from voice_ai_banking_support_agent.runtime.orchestrator import RuntimeOrchestrator, RuntimeRequest
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


def test_runtime_smoke_answered() -> None:
    orch = RuntimeOrchestrator(retriever=FakeRetriever())  # type: ignore[arg-type]
    store = SessionStateStore()
    state = store.get_or_create("s1")
    out = orch.handle(
        RuntimeRequest(session_id="s1", query="Ի՞նչ ավանդներ կան Ամերիայում", index_name="i", verbose=True), state
    )
    assert out.status == "answered"
    assert out.detected_topic == "deposit"
    assert out.decision_trace


def test_runtime_smoke_refused_unsupported() -> None:
    orch = RuntimeOrchestrator(retriever=FakeRetriever())  # type: ignore[arg-type]
    store = SessionStateStore()
    state = store.get_or_create("s2")
    out = orch.handle(RuntimeRequest(session_id="s2", query="Which bank is best for loans?", index_name="i"), state)
    assert out.status == "refused"
    assert out.refusal_reason == "unsupported_request_type"

