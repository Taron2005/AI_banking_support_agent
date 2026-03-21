from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.runtime.evidence_checker import EvidenceChecker
from voice_ai_banking_support_agent.runtime.models import RetrievedChunk
from voice_ai_banking_support_agent.runtime.refusal import refusal_message


def _chunk(text: str, topic: str = "branch") -> RetrievedChunk:
    return RetrievedChunk(
        score=0.5,
        chunk=DocumentMetadata(
            bank_key="idbank",
            bank_name="IDBank",
            topic=topic,  # type: ignore[arg-type]
            source_url="https://example.am",
            page_title="p",
            section_title="s",
            language="hy",
            chunk_id="c1",
            raw_text=text,
            cleaned_text=text,
        ),
    )


def test_refusal_messages_available() -> None:
    assert "Կներեք" in refusal_message("out_of_scope")


def test_evidence_checker_branch_address_requirement() -> None:
    checker = EvidenceChecker()
    out = checker.assess("որտե՞ղ է մասնաճյուղը", "branch", [_chunk("մասնաճյուղի տվյալներ")])
    assert out.sufficient is False
    out_ok = checker.assess("որտե՞ղ է մասնաճյուղը", "branch", [_chunk("Հասցե: Երևան, ...")])
    assert out_ok.sufficient is True

