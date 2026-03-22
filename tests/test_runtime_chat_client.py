from voice_ai_banking_support_agent.runtime.models import RuntimeResponse
from voice_ai_banking_support_agent.runtime.orchestrator import RuntimeRequest
from voice_ai_banking_support_agent.voice.runtime_chat_client import (
    runtime_response_from_chat_payload,
)


def test_runtime_response_from_chat_payload_strips_extra_keys() -> None:
    body = {
        "answer_text": "Պատասխան",
        "status": "answered",
        "refusal_reason": None,
        "answer_synthesis": "llm",
        "llm_error": None,
        "detected_topic": "deposit",
        "detected_bank": "idbank",
        "detected_banks": ["idbank"],
        "used_sources": [],
        "retrieved_chunks_summary": [],
        "state_updates": {},
        "decision_trace": [],
        "llm_provider": "gemini",
        "llm_model": "gemini-2.0-flash",
    }
    rr = runtime_response_from_chat_payload(body)
    assert isinstance(rr, RuntimeResponse)
    assert rr.answer_text == "Պատասխան"
    assert rr.status == "answered"


def test_repair_stt_transcript_fixes_common_bank_typos() -> None:
    from voice_ai_banking_support_agent.runtime.query_normalizer import repair_stt_transcript

    assert "ամերիաբանկ" in repair_stt_transcript("ինչ ավանդներ ունի ամերիբանկը")
