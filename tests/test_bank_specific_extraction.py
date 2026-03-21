from voice_ai_banking_support_agent.extraction.cleaning import clean_html_to_text
from voice_ai_banking_support_agent.extraction.section_parser import parse_sections_from_html
from voice_ai_banking_support_agent.scrapers.acba import AcbaScraper
from voice_ai_banking_support_agent.scrapers.ameriabank import AmeriaBankScraper
from voice_ai_banking_support_agent.scrapers.idbank import IDBankScraper


def test_scrapers_define_selector_rationales():
    for scraper in [AcbaScraper(), AmeriaBankScraper(), IDBankScraper()]:
        rules = scraper.extraction_rules()
        assert rules.remove_selectors
        assert rules.prefer_content_selectors
        assert rules.fallback_block_selectors

        for rule in rules.remove_selectors + rules.prefer_content_selectors + rules.fallback_block_selectors:
            assert rule.selector.strip()
            assert rule.why.strip()


def test_bank_specific_selector_removes_nav_noise():
    html = """
    <html><body>
      <div class="main-menu">MENU SHOULD BE REMOVED</div>
      <main>
        <h1>Վարկեր</h1>
        <p>Վարկի պայմանների հիմնական նկարագրություն և տոկոսադրույքների տեղեկություն:</p>
      </main>
      <footer>FOOTER SHOULD BE REMOVED</footer>
    </body></html>
    """
    rules = AcbaScraper().extraction_rules()
    cleaned = clean_html_to_text(html, rules=rules)
    assert "MENU SHOULD BE REMOVED" not in cleaned.cleaned_text
    assert "FOOTER SHOULD BE REMOVED" not in cleaned.cleaned_text
    assert "Վարկի պայմանների" in cleaned.cleaned_text


def test_fallback_block_parsing_when_headings_absent():
    html = """
    <html><body>
      <div class="branch-item">
        <div class="title">Արաբկիր մասնաճյուղ</div>
        <div>56/162 Կոմիտաս պողոտա</div>
        <div>Երկ-Ուրբ 09:15-18:45</div>
      </div>
      <div class="branch-item">
        <div class="title">Կենտրոն մասնաճյուղ</div>
        <div>6 Northern Ave.</div>
        <div>Երկ-Շբթ 10:15-21:30</div>
      </div>
    </body></html>
    """
    rules = IDBankScraper().extraction_rules()
    sections = parse_sections_from_html(html, rules=rules, min_content_chars=20)
    assert len(sections) >= 2
    assert any("Արաբկիր" in s.content_text for s in sections)
    assert any("Կենտրոն" in s.content_text for s in sections)

