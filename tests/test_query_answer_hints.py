from voice_ai_banking_support_agent.runtime.bank_scope import query_implies_all_banks
from voice_ai_banking_support_agent.runtime.evidence_checker import EvidenceChecker, EvidenceConfig
from voice_ai_banking_support_agent.runtime.query_answer_hints import (
    extra_llm_context,
    retrieval_query_with_topic_boost,
)
from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk


def test_retrieval_boost_adds_term_deposit_tokens() -> None:
    q = "Համեմատիր Ամերիաբանկի և Իդբանկի ժամկետային ավանդների տոկոսները"
    out = retrieval_query_with_topic_boost(q, runtime_topic="deposit")
    assert "ժամկետային" in out
    assert "ֆիքսված" in out or "ժամկետ" in out


def test_extra_llm_context_deposit_compare_general() -> None:
    t = extra_llm_context("Ով է ավելի բարձր տոկոս ավանդի համար Ameriabank թե IDBank", runtime_topic="deposit")
    assert t
    assert "ժամկետային" in t or "ցպահանջ" in t


def test_query_implies_all_banks_for_bolor_avand() -> None:
    assert query_implies_all_banks("Տուր հղումներ բոլոր ավանդների էջերի համար".lower())


def test_branch_evidence_accepts_listing_without_hase() -> None:
    body = "«Կենտրոն» մասնաճյուղ  ՀՀ, ք. Երևան, Հյուսիսային պող., շենք 6"
    chk = EvidenceChecker(
        EvidenceConfig(min_top_score=0.05, branch_address_patterns=("հասցե",))
    )
    chunk = RetrievedChunk(
        score=0.5,
        chunk=DocumentMetadata(
            bank_key="idbank",
            bank_name="IDBank",
            topic="branch",
            source_url="https://idbank.am/br",
            page_title="B",
            section_title="S",
            language="hy",
            chunk_id="z",
            raw_text="r",
            cleaned_text=body,
        ),
    )
    d = chk.assess("Իդբանկի մասնաճյուղերի հասցեները", "branch", [chunk])
    assert d.sufficient
