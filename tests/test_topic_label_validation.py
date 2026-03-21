import pytest
from pydantic import ValidationError

from voice_ai_banking_support_agent.models import DocumentMetadata


def _make_doc(topic: str) -> DocumentMetadata:
    return DocumentMetadata(
        bank_key="acba",
        bank_name="ACBA Bank",
        topic=topic,  # type: ignore[arg-type]
        source_url="https://example.invalid/page",
        page_title="Test Page",
        section_title="Test Section",
        language="hy",
        chunk_id="chunk_1",
        raw_text="raw",
        cleaned_text="cleaned",
    )


def test_topic_label_validation_accepts_allowed_values():
    for topic in ["credit", "deposit", "branch"]:
        doc = _make_doc(topic)
        assert doc.topic == topic


def test_topic_label_validation_rejects_unknown_values():
    with pytest.raises(ValidationError):
        _make_doc("unknown")

