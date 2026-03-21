from voice_ai_banking_support_agent.config import ChunkingConfig
from voice_ai_banking_support_agent.extraction.section_parser import Section
from voice_ai_banking_support_agent.indexing.chunker import chunk_sections


def test_chunker_preserves_section_boundaries_and_headings():
    sections = [
        Section(
            title="Վարկեր",
            level=1,
            content_text=(
                "Վարկի տոկոսադրույք՝ 12.5%։ Գործող ժամկետ՝ 36 ամիս։ "
                "Մանրամասները ներկայացված են ստորև։"
            ),
        ),
        Section(
            title="Ավանդներ",
            level=1,
            content_text=(
                "Ավանդի տոկոսադրույք՝ 7.0%։ Գործող ժամկետ՝ 12 ամիս։ "
                "Հնարավոր է լրացուցիչ համալրում։"
            ),
        ),
    ]

    docs = chunk_sections(
        sections=sections,
        bank_key="acba",
        bank_name="ACBA Bank",
        topic="credit",  # type: ignore[arg-type]
        source_url="https://example.invalid/credits",
        page_title="Test",
        language="hy",
        raw_page_text="raw",
        chunking=ChunkingConfig(target_words=10, min_words=3, max_words=40),
    )

    assert len(docs) > 0
    for d in docs:
        assert d.bank_key == "acba"
        assert d.section_title in {"Վարկեր", "Ավանդներ"}
        # Chunk text should start with the section title for retrieval grounding.
        assert d.cleaned_text.startswith(d.section_title + "\n")
        other_heading = "Ավանդներ" if d.section_title == "Վարկեր" else "Վարկեր"
        assert other_heading not in d.cleaned_text

    # Ensure both headings produced at least one chunk.
    produced = {d.section_title for d in docs}
    assert produced == {"Վարկեր", "Ավանդներ"}

