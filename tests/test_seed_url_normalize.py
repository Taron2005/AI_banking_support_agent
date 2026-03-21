from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from voice_ai_banking_support_agent.config import NetworkConfig
from voice_ai_banking_support_agent.scrapers.base import RequestsHTMLFetcher, normalize_seed_url


def test_normalize_seed_url_fixes_acba_duplicate_path_segments() -> None:
    raw = "https://www.acba.am/hy/individuals/hy/individuals/loans/consumer-credits"
    fixed = "https://www.acba.am/hy/individuals/loans/consumer-credits"
    assert normalize_seed_url(raw) == fixed


def test_normalize_seed_url_idempotent_for_clean_acba_url() -> None:
    u = "https://www.acba.am/hy/individuals/loans"
    assert normalize_seed_url(u) == u


def test_normalize_seed_url_leaves_other_banks_unchanged() -> None:
    u = "https://idbank.am/credits/"
    assert normalize_seed_url(u) == u


def test_fetch_does_not_retry_http_404() -> None:
    fetcher = RequestsHTMLFetcher(NetworkConfig(retries=5, backoff_min_seconds=0.0, backoff_max_seconds=0.0))
    resp = MagicMock()
    resp.status_code = 404
    resp.headers = {"Content-Type": "text/html; charset=utf-8"}
    resp.url = "https://example.test/missing"
    resp.content = b"<html>not found</html>"
    resp.apparent_encoding = "utf-8"
    fetcher._session.get = MagicMock(return_value=resp)
    with pytest.raises(RuntimeError, match="404"):
        fetcher.fetch("https://example.test/missing")
    assert fetcher._session.get.call_count == 1


def test_fetch_retries_then_succeeds_on_http_500() -> None:
    fetcher = RequestsHTMLFetcher(
        NetworkConfig(retries=3, backoff_min_seconds=0.0, backoff_max_seconds=0.0, timeout_seconds=5.0)
    )
    bad = MagicMock()
    bad.status_code = 500
    bad.headers = {"Content-Type": "text/html; charset=utf-8"}
    bad.url = "https://example.test/flaky"
    bad.content = b"err"
    good = MagicMock()
    good.status_code = 200
    good.headers = {"Content-Type": "text/html; charset=utf-8"}
    good.url = "https://example.test/flaky"
    good.content = b"<html>" + (b"x" * 60) + b"</html>"
    good.apparent_encoding = "utf-8"
    fetcher._session.get = MagicMock(side_effect=[bad, good])
    out = fetcher.fetch("https://example.test/flaky")
    assert "html" in out.html.lower()
    assert fetcher._session.get.call_count == 2
