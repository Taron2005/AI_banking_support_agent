"""Streaming runtime: ``stream_handle`` must mirror gates and end with one terminal response."""

from voice_ai_banking_support_agent.runtime.answer_generator import (
    AnswerGeneratorConfig,
    GroundedAnswerGenerator,
    LLMAnswerGenerator,
)
from voice_ai_banking_support_agent.runtime.llm import MockLLMClient
from voice_ai_banking_support_agent.runtime.orchestrator import RuntimeOrchestrator, RuntimeRequest
from voice_ai_banking_support_agent.runtime.session_state import SessionStateStore

from test_runtime_smoke import FakeRetriever


def test_stream_handle_llm_emits_deltas_then_single_done() -> None:
    llm = MockLLMClient()
    gen = LLMAnswerGenerator(
        llm_client=llm,
        fallback=GroundedAnswerGenerator(),
        cfg=AnswerGeneratorConfig(),
    )
    orch = RuntimeOrchestrator(retriever=FakeRetriever(), answer_generator=gen)  # type: ignore[arg-type]
    store = SessionStateStore()
    state = store.get_or_create("stream-1")
    parts = list(
        orch.stream_handle(
            RuntimeRequest(
                session_id="stream-1",
                query="Ի՞նչ ավանդներ կան Ամերիայում",
                index_name="i",
            ),
            state,
        )
    )
    deltas = [p.text_delta for p in parts if p.text_delta]
    finals = [p.done for p in parts if p.done is not None]
    assert len(deltas) >= 1
    assert len(finals) == 1
    assert finals[0].status == "answered"


def test_stream_handle_refusal_is_single_done_chunk() -> None:
    llm = MockLLMClient()
    gen = LLMAnswerGenerator(
        llm_client=llm,
        fallback=GroundedAnswerGenerator(),
        cfg=AnswerGeneratorConfig(),
    )
    orch = RuntimeOrchestrator(retriever=FakeRetriever(), answer_generator=gen)  # type: ignore[arg-type]
    store = SessionStateStore()
    state = store.get_or_create("stream-2")
    parts = list(
        orch.stream_handle(
            RuntimeRequest(session_id="stream-2", query="Which bank is best for loans?", index_name="i"),
            state,
        )
    )
    assert len(parts) == 1
    assert parts[0].done is not None
    assert parts[0].done.status == "refused"
