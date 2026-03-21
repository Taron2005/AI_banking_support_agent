from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk
from voice_ai_banking_support_agent.runtime.orchestrator import RuntimeOrchestrator, RuntimeRequest
from voice_ai_banking_support_agent.runtime.runtime_config import OrchestrationSettings
from voice_ai_banking_support_agent.runtime.session_state import SessionStateStore


class _FakeRetriever:
    def retrieve(self, req):  # noqa: ANN001
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


def test_require_explicit_bank_triggers_clarify_without_bank_name() -> None:
    orch = RuntimeOrchestrator(
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        orchestration=OrchestrationSettings(require_explicit_bank=True),
        bank_aliases={"acba": ["acba"], "ameriabank": ["ameria"], "idbank": ["idbank"]},
    )
    st = SessionStateStore().get_or_create("x")
    out = orch.handle(
        RuntimeRequest(session_id="x", query="Ինչ ավանդներ կան հիմա", index_name="i"),
        st,
    )
    assert out.status == "clarify"
    assert "բանկ" in out.answer_text.lower()


def test_require_explicit_bank_skipped_when_all_banks_phrase() -> None:
    orch = RuntimeOrchestrator(
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        orchestration=OrchestrationSettings(require_explicit_bank=True),
        bank_aliases={"acba": ["acba"], "ameriabank": ["ameria"], "idbank": ["idbank"]},
    )
    st = SessionStateStore().get_or_create("y")
    out = orch.handle(
        RuntimeRequest(session_id="y", query="Ինչ ավանդներ կան բոլոր բանկերում", index_name="i"),
        st,
    )
    assert out.status == "answered"
