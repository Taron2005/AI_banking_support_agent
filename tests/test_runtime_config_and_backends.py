from pathlib import Path

from voice_ai_banking_support_agent.runtime.answer_generator import GroundedAnswerGenerator, LLMAnswerGenerator
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk
from voice_ai_banking_support_agent.runtime.runtime_config import load_runtime_settings
from voice_ai_banking_support_agent.models import DocumentMetadata


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        score=0.9,
        chunk=DocumentMetadata(
            bank_key="acba",
            bank_name="ACBA Bank",
            topic="deposit",
            source_url="https://acba.am/deposits",
            page_title="Deposits",
            section_title="Classic",
            language="hy",
            chunk_id="c1",
            raw_text="raw",
            cleaned_text="Ավանդի տոկոսադրույք մինչև 8%",
        ),
    )


def test_runtime_config_loader_defaults() -> None:
    cfg = load_runtime_settings(None)
    assert cfg.retrieval.default_top_k >= 1
    assert "ameriabank" in cfg.bank_aliases


def test_runtime_config_loader_yaml(tmp_path: Path) -> None:
    p = tmp_path / "runtime.yaml"
    p.write_text("retrieval:\n  default_top_k: 9\nanswer:\n  backend: extractive\n", encoding="utf-8")
    cfg = load_runtime_settings(p)
    assert cfg.retrieval.default_top_k == 9


def test_llm_backend_falls_back_when_no_client() -> None:
    fallback = GroundedAnswerGenerator()
    llm = LLMAnswerGenerator(llm_client=None, fallback=fallback)
    out = llm.generate("Ի՞նչ ավանդներ կան", "deposit", [_chunk()], frozenset({"acba"}))
    assert "պաշտոնական" in out or "նիշքած" in out or "ամփոփում" in out

