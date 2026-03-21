from voice_ai_banking_support_agent.scrapers.acba import AcbaScraper
from voice_ai_banking_support_agent.scrapers.ameriabank import AmeriaBankScraper
from voice_ai_banking_support_agent.scrapers.base import RequestsHTMLFetcher
from voice_ai_banking_support_agent.scrapers.idbank import IDBankScraper
from voice_ai_banking_support_agent.config import NetworkConfig


def _fetcher() -> RequestsHTMLFetcher:
    return RequestsHTMLFetcher(NetworkConfig(timeout_seconds=2, retries=1))


def test_each_scraper_has_structured_fetch_method() -> None:
    html = "<html><head><title>t</title></head><body><a href='/service-network'>x</a></body></html>"
    for scraper in [AcbaScraper(), AmeriaBankScraper(), IDBankScraper()]:
        result = scraper.fetch_structured(
            fetcher=_fetcher(),
            url="https://example.am/service-network",
            html=html,
            topic="branch",
        )
        assert isinstance(result.branch_records, list)
        assert isinstance(result.discovered_urls, list)
        assert isinstance(result.notes, list)


def test_ameriabank_dom_fallback_produces_branch_record() -> None:
    html = """
    <html><body>
      <script>window.__SOMETHING = {"x": 1};</script>
      <div>Վազգեն Սարգսյան 2, Երևան 0010, ՀՀ</div>
      <div>(+37410) 56 11 11</div>
    </body></html>
    """
    result = AmeriaBankScraper().fetch_structured(
        fetcher=_fetcher(),
        url="https://ameriabank.am/service-network",
        html=html,
        topic="branch",
    )
    assert result.branch_records
    assert any("Երևան" in r.address for r in result.branch_records)


def test_ameriabank_fetches_dnn_module_payload_when_init_modules_present() -> None:
    class DummyFetcher:
        def fetch_json(self, *_args, **_kwargs):
            return (
                200,
                {
                    "status": "success",
                    "data": {
                        "slides": [
                            {
                                "description": {
                                    "original": (
                                        "<div>Ամերիա Կուտակային ավանդ</div>"
                                        "<div>Տոկոսադրույք մինչև 8%</div>"
                                    )
                                }
                            }
                        ]
                    },
                },
            )

    html = """
    <html><body>
      <input name="__RequestVerificationToken" type="hidden" value="token123" />
      <script>
        var args = { tabId: 12077 };
        (window.wscMCMModuleManagerEdit || window.wscMCMModuleManagerView).initModule(50811, args);
      </script>
    </body></html>
    """
    result = AmeriaBankScraper().fetch_structured(
        fetcher=DummyFetcher(),  # type: ignore[arg-type]
        url="https://ameriabank.am/personal/saving/deposits/cumulative-deposit",
        html=html,
        topic="deposit",
    )
    assert result.supplemental_html is not None
    assert "Կուտակային" in result.supplemental_html
