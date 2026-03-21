from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import NetworkConfig
from ..models import BranchRecord

logger = logging.getLogger(__name__)


class TransientFetchError(Exception):
    """Raised for connection/timeouts and HTTP 5xx so tenacity can retry without spinning on 404/403."""


_CHARSET_RE = re.compile(r"charset=([\w._-]+)", re.IGNORECASE)


def normalize_seed_url(url: str) -> str:
    """
    Normalize known-broken seed URL shapes before fetch and stable_id hashing.

    ACBA occasionally had duplicated `/hy/individuals/` path segments in manifests;
    the duplicate form often still responds 200 but should not be treated as canonical.
    """

    parsed = urlparse(url.strip())
    if "acba.am" not in parsed.netloc.lower():
        return url.strip()
    path = parsed.path
    while "/hy/individuals/hy/individuals/" in path:
        path = path.replace("/hy/individuals/hy/individuals/", "/hy/individuals/", 1)
    if path == parsed.path:
        return url.strip()
    fixed = urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, parsed.fragment))
    logger.debug("Normalized ACBA duplicate path segment url=%s -> %s", url.strip(), fixed)
    return fixed


def _charset_from_content_type(content_type: str) -> str | None:
    m = _CHARSET_RE.search(content_type or "")
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")


def _decode_html_bytes(raw: bytes, content_type: str, apparent: str | None, url: str) -> str:
    """
    Decode HTML bytes explicitly (do not rely on `response.text` alone).

    Order: declared charset → UTF-8 strict → requests apparent_encoding → UTF-8 replace (last resort).
    """

    declared = _charset_from_content_type(content_type)
    candidates: list[str] = []
    if declared:
        candidates.append(declared)
    candidates.append("utf-8")
    if apparent and apparent.lower() not in {c.lower() for c in candidates}:
        candidates.append(apparent)
    candidates.append("cp1256")
    candidates.append("windows-1252")

    last_err: UnicodeDecodeError | None = None
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError) as e:
            if isinstance(e, UnicodeDecodeError):
                last_err = e
            continue

    text = raw.decode("utf-8", errors="replace")
    logger.warning(
        "HTML decode used UTF-8 replacement fallback url=%s last_error=%s", url, last_err
    )
    return text


def _should_retry_exception(exc: BaseException) -> bool:
    if isinstance(exc, TransientFetchError):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.ChunkedEncodingError):
        return True
    return False


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

    def _maybe_throttle(self) -> None:
        delay = self._network.request_delay_seconds
        if delay and delay > 0:
            time.sleep(delay)

    def fetch(self, url: str) -> HTMLFetchResult:
        """
        Fetch HTML from `url`.

        Retries transient failures only (timeouts, connection errors, HTTP 5xx).
        Client errors (4xx) and bad content type fail immediately.
        """

        headers = {"User-Agent": self._network.user_agent, "Accept-Language": "hy,en;q=0.9,en;q=0.8"}
        timeout = self._network.timeout_seconds

        def _do_fetch() -> HTMLFetchResult:
            self._maybe_throttle()
            try:
                resp = self._session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            except requests.RequestException as e:
                raise TransientFetchError(str(e)) from e

            content_type = (resp.headers.get("Content-Type") or "").lower()
            if resp.status_code >= 500:
                raise TransientFetchError(f"HTTP {resp.status_code} for url={url}")
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP fetch failed status={resp.status_code} url={url} final_url={resp.url}")
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                raise RuntimeError(
                    f"Unexpected content type={content_type!r} for url={url}. "
                    "Expected HTML page."
                )

            apparent = getattr(resp, "apparent_encoding", None)
            html = _decode_html_bytes(resp.content, content_type, apparent, url)
            if not html or len(html.strip()) < 50:
                raise RuntimeError(
                    f"Empty/too-short HTML response for url={url} final_url={resp.url} status={resp.status_code}"
                )
            bad_chars = html.count("\ufffd")
            if bad_chars > max(8, len(html) // 2000):
                raise RuntimeError(
                    f"Garbled HTML ({bad_chars} U+FFFD replacement chars) for url={url} final_url={resp.url}. "
                    "Check charset / encoding."
                )

            logger.info(
                "Fetched html url=%s final_url=%s status=%s content_type=%s chars=%d",
                url,
                resp.url,
                resp.status_code,
                (resp.headers.get("Content-Type") or "").split(";")[0].strip(),
                len(html),
            )
            return HTMLFetchResult(
                url=url,
                status_code=resp.status_code,
                html=html,
                final_url=resp.url,
            )

        retrying = Retrying(
            reraise=True,
            retry=retry_if_exception(_should_retry_exception),
            stop=stop_after_attempt(max(1, self._network.retries)),
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
            self._maybe_throttle()
            try:
                resp = self._session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json_body,
                    headers=merged_headers,
                    timeout=timeout,
                    allow_redirects=True,
                )
            except requests.RequestException as e:
                raise TransientFetchError(str(e)) from e

            if resp.status_code >= 500:
                raise TransientFetchError(f"HTTP {resp.status_code} for url={url}")

            payload: dict[str, Any] | list[Any] | None = None
            try:
                payload = resp.json()
            except ValueError:
                payload = None

            ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
            logger.debug(
                "JSON request url=%s method=%s status=%s content_type=%s",
                url,
                method.upper(),
                resp.status_code,
                ct,
            )
            return resp.status_code, payload

        retrying = Retrying(
            reraise=True,
            retry=retry_if_exception(_should_retry_exception),
            stop=stop_after_attempt(max(1, self._network.retries)),
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
        absolute = normalize_seed_url(absolute)
        parsed = urlparse(absolute)
        if parsed.netloc != base.netloc:
            continue
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        normalized = normalize_seed_url(normalized)
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

