from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.answer_generator import GroundedAnswerGenerator, LLMAnswerGenerator
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk


class _GoodClient:
    def generate(self, prompt: str) -> str:
        return "Սա LLM պատասխան է։"


class _BadClient:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("boom")


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        score=0.7,
        chunk=DocumentMetadata(
            bank_key="acba",
            bank_name="ACBA Bank",
            topic="deposit",
            source_url="https://acba.am",
            page_title="p",
            section_title="s",
            language="hy",
            chunk_id="c1",
            raw_text="raw",
            cleaned_text="Ավանդի պայմաններ",
        ),
    )


def test_llm_backend_uses_client_when_available() -> None:
    gen = LLMAnswerGenerator(llm_client=_GoodClient(), fallback=GroundedAnswerGenerator())
    out = gen.generate("q", "deposit", [_chunk()], "acba")
    assert "LLM" in out


def test_llm_backend_falls_back_on_error() -> None:
    gen = LLMAnswerGenerator(llm_client=_BadClient(), fallback=GroundedAnswerGenerator())
    out = gen.generate("q", "deposit", [_chunk()], "acba")
    assert "Հիմնական տվյալներ" in out

