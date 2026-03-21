from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.answer_generator import GroundedAnswerGenerator, LLMAnswerGenerator
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk


class _GoodClient:
    def generate(self, prompt: str) -> str:
        return "Սա LLM պատասխան է։"


class _BadClient:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("boom")


class _HallucinatedUrlClient:
    def generate(self, prompt: str) -> str:
        return (
            "Տեքստ։ https://evil.example/not-in-evidence\n\n"
            "Աղբյուրներ։\n"
            "https://acba.am/deposits"
        )


class _MarkdownEvilLinkClient:
    def generate(self, prompt: str) -> str:
        return "Տեքստ [կայք](https://evil.example/x) և [ok](https://acba.am/deposits/extra)."


class _EchoPolicyClient:
    def generate(self, prompt: str) -> str:
        return (
            "Մի օգտագործիր արտաքին գիտելիք։\n\n"
            "Համառոտ ամփոփում՝\n"
            "Ավանդների մասին։\n"
        )


class _CapturePromptClient:
    def __init__(self) -> None:
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "ok"


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        score=0.7,
        chunk=DocumentMetadata(
            bank_key="acba",
            bank_name="ACBA Bank",
            topic="deposit",
            source_url="https://acba.am/deposits",
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
    out = gen.generate("q", "deposit", [_chunk()], frozenset({"acba"}))
    assert "LLM" in out


def test_llm_generate_answer_result_marks_llm_source() -> None:
    gen = LLMAnswerGenerator(llm_client=_GoodClient(), fallback=GroundedAnswerGenerator())
    ar = gen.generate_answer_result("q", "deposit", [_chunk()], frozenset({"acba"}))
    assert ar.answer_synthesis == "llm"
    assert ar.llm_error is None
    assert "LLM" in ar.text


def test_llm_generate_answer_result_marks_fallback_on_error() -> None:
    gen = LLMAnswerGenerator(llm_client=_BadClient(), fallback=GroundedAnswerGenerator())
    ar = gen.generate_answer_result("q", "deposit", [_chunk()], frozenset({"acba"}))
    assert ar.answer_synthesis == "extractive_fallback"
    assert ar.llm_error is not None


def test_llm_backend_falls_back_on_error() -> None:
    gen = LLMAnswerGenerator(llm_client=_BadClient(), fallback=GroundedAnswerGenerator())
    out = gen.generate("q", "deposit", [_chunk()], frozenset({"acba"}))
    assert "պաշտոնական" in out or "նիշքած" in out or "ամփոփում" in out


def test_llm_backend_strips_urls_not_in_evidence() -> None:
    gen = LLMAnswerGenerator(llm_client=_HallucinatedUrlClient(), fallback=GroundedAnswerGenerator())
    out = gen.generate("q", "deposit", [_chunk()], frozenset({"acba"}))
    assert "evil.example" not in out
    assert "acba.am" in out


def test_llm_backend_strips_markdown_links_not_in_evidence() -> None:
    gen = LLMAnswerGenerator(llm_client=_MarkdownEvilLinkClient(), fallback=GroundedAnswerGenerator())
    out = gen.generate("q", "deposit", [_chunk()], frozenset({"acba"}))
    assert "evil.example" not in out
    assert "կայք" in out
    assert "acba.am" in out


def test_llm_success_appends_footnote_when_missing() -> None:
    gen = LLMAnswerGenerator(llm_client=_GoodClient(), fallback=GroundedAnswerGenerator())
    ar = gen.generate_answer_result("q", "deposit", [_chunk()], frozenset({"acba"}))
    assert ar.answer_synthesis == "llm"
    assert "Նշում․" in ar.text
    assert "արհեստական բանականությամբ" in ar.text


def test_llm_backend_strips_echoed_policy_lines() -> None:
    gen = LLMAnswerGenerator(llm_client=_EchoPolicyClient(), fallback=GroundedAnswerGenerator())
    out = gen.generate("q", "deposit", [_chunk()], frozenset({"acba"}))
    assert "արտաքին գիտելիք" not in out
    assert "Ավանդ" in out


def test_llm_backend_filters_evidence_by_selected_bank() -> None:
    id_chunk = RetrievedChunk(
        score=0.9,
        chunk=DocumentMetadata(
            bank_key="idbank",
            bank_name="IDBank",
            topic="deposit",
            source_url="https://idbank.am/x",
            page_title="p",
            section_title="s",
            language="hy",
            chunk_id="c2",
            raw_text="raw",
            cleaned_text="IDBank տեքստ",
        ),
    )
    cap = _CapturePromptClient()
    gen = LLMAnswerGenerator(llm_client=cap, fallback=GroundedAnswerGenerator())
    gen.generate("q", "deposit", [_chunk(), id_chunk], frozenset({"acba"}))
    assert "idbank.am" not in cap.last_prompt
    assert "acba.am" in cap.last_prompt

