from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import Retrying, stop_after_attempt, wait_exponential

from ..config import NetworkConfig
from ..models import BranchRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HTMLFetchResult:
    """Result of an offline HTML fetch."""

    url: str
    status_code: int
    html: str
    final_url: str


@dataclass(frozen=True)
class SelectorRule:
    """
    One CSS selector rule with a human-readable rationale.

    We keep rationale near selectors so future maintainers know exactly why
    each bank-specific rule exists and can update it safely when markup changes.
    """

    selector: str
    why: str


@dataclass(frozen=True)
class BankExtractionRules:
    """
    Bank-specific extraction controls used by cleaning/section parsers.

    Fields:
    - remove_selectors: aggressively remove noisy UI components.
    - prefer_content_selectors: prioritize these content containers when present.
    - fallback_block_selectors: if heading parsing fails, parse repeated blocks/cards.
    """

    remove_selectors: list[SelectorRule]
    prefer_content_selectors: list[SelectorRule]
    fallback_block_selectors: list[SelectorRule]


@dataclass(frozen=True)
class StructuredFetchResult:
    """
    Bank-specific structured extraction output.

    `branch_records` is currently the primary output for branch pages.
    `discovered_urls` lets scrapers suggest extra in-domain candidate URLs.
    `notes` contains debug-friendly messages describing what strategy worked.
    """

    branch_records: list[BranchRecord]
    discovered_urls: list[str]
    notes: list[str]
    supplemental_html: str | None = None


class RequestsHTMLFetcher:
    """Fetch HTML from a URL using `requests` with retries."""

    def __init__(self, network: NetworkConfig) -> None:
        self._network = network
        self._session = requests.Session()

    def fetch(self, url: str) -> HTMLFetchResult:
        """
        Fetch HTML from `url`.

        Retries are handled by `tenacity`; network config controls timeout, UA, and retry/backoff.
        """

        headers = {"User-Agent": self._network.user_agent, "Accept-Language": "hy,en;q=0.9,en;q=0.8"}
        timeout = self._network.timeout_seconds

        def _do_fetch() -> HTMLFetchResult:
            resp = self._session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            resp.encoding = resp.encoding or "utf-8"
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP fetch failed status={resp.status_code} url={url}")
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                raise RuntimeError(
                    f"Unexpected content type={content_type!r} for url={url}. "
                    "Expected HTML page."
                )
            if not resp.text or len(resp.text.strip()) < 50:
                raise RuntimeError(f"Empty/too-short HTML response for url={url}")
            return HTMLFetchResult(
                url=url,
                status_code=resp.status_code,
                html=resp.text,
                final_url=resp.url,
            )

        # Use Retrying so we can bind instance config values (no static decorator args).
        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(self._network.retries),
            wait=wait_exponential(
                min=self._network.backoff_min_seconds, max=self._network.backoff_max_seconds
            ),
        )
        for attempt in retrying:
            with attempt:
                return _do_fetch()
        raise RuntimeError("Unreachable: Retrying loop ended unexpectedly.")

    def fetch_json(
        self,
        url: str,
        *,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any] | list[Any] | None]:
        """Best-effort JSON fetch helper for bank-specific API probing."""

        merged_headers = {
            "User-Agent": self._network.user_agent,
            "Accept-Language": "hy,en;q=0.9,en;q=0.8",
            "Accept": "application/json, text/plain, */*",
        }
        if headers:
            merged_headers.update(headers)
        timeout = self._network.timeout_seconds

        def _do_fetch() -> tuple[int, dict[str, Any] | list[Any] | None]:
            resp = self._session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                headers=merged_headers,
                timeout=timeout,
                allow_redirects=True,
            )
            payload: dict[str, Any] | list[Any] | None = None
            try:
                payload = resp.json()
            except ValueError:
                payload = None
            return resp.status_code, payload

        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(self._network.retries),
            wait=wait_exponential(
                min=self._network.backoff_min_seconds, max=self._network.backoff_max_seconds
            ),
        )
        for attempt in retrying:
            with attempt:
                return _do_fetch()
        raise RuntimeError("Unreachable: Retrying loop ended unexpectedly.")


def parse_json_ld_objects(html: str) -> list[dict[str, Any]]:
    """Extract JSON-LD objects from page scripts."""

    soup = BeautifulSoup(html, "lxml")
    out: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        text = script.string or script.get_text() or ""
        if not text.strip():
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
        elif isinstance(obj, list):
            out.extend([x for x in obj if isinstance(x, dict)])
    return out


def parse_inline_json_objects(html: str) -> list[dict[str, Any]]:
    """
    Extract inline JS-assigned JSON blocks.

    Targets patterns like `window.__INITIAL_STATE__ = {...};`.
    """

    blocks = re.findall(
        r"window\.[A-Za-z0-9_]+\s*=\s*(\{[\s\S]{50,}?\})\s*;",
        html,
        flags=re.MULTILINE,
    )
    out: list[dict[str, Any]] = []
    for raw in blocks:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def extract_same_domain_links(html: str, base_url: str) -> list[str]:
    """Collect de-duplicated absolute links from the same domain."""

    soup = BeautifulSoup(html, "lxml")
    base = urlparse(base_url)
    links: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc != base.netloc:
            continue
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
    return links


def extract_page_title(html: str) -> str:
    """Extract a best-effort page title from `<title>` or first `h1`."""

    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    h1 = soup.find("h1")
    if h1 is not None:
        return h1.get_text(strip=True)
    return ""

