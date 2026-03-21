from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from ..bank_manifest import load_banks_manifest
from ..config import AppConfig
from ..scrapers.base import RequestsHTMLFetcher, extract_same_domain_links, normalize_seed_url
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _classify_topics(url: str, text_hint: str) -> list[str]:
    u = url.lower()
    t = text_hint.lower()
    topics: set[str] = set()

    credit_markers = ["credit", "loan", "վարկ", "վարկեր"]
    deposit_markers = ["deposit", "saving", "ավանդ", "խնայ"]
    branch_markers = ["branch", "atm", "service-network", "մասնաճյուղ", "բանկոմատ"]

    if any(m in u or m in t for m in credit_markers):
        topics.add("credit")
    if any(m in u or m in t for m in deposit_markers):
        topics.add("deposit")
    if any(m in u or m in t for m in branch_markers):
        topics.add("branch")
    return sorted(topics)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_urls(
    *,
    config: AppConfig,
    bank_keys: list[str] | None,
    max_pages: int = 120,
    max_depth: int = 2,
) -> None:
    """
    Controlled same-domain URL discovery for manual review.

    This mode does NOT auto-ingest discovered links. It only writes review files:
    `data/discovery/<bank_key>_url_candidates.jsonl`.
    """

    setup_logging()
    config.ensure_dirs()

    manifest = load_banks_manifest(config.manifest_path)
    fetcher = RequestsHTMLFetcher(config.network)

    wanted = {b.lower() for b in bank_keys} if bank_keys else None
    discovery_dir = config.data_dir / "discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)

    for bank in manifest.banks:
        if wanted and bank.bank_key.lower() not in wanted:
            continue

        raw_seeds = {*bank.credits.urls, *bank.deposits.urls, *bank.branches.urls}
        seeds: list[str] = []
        seen_seed: set[str] = set()
        for u in raw_seeds:
            nu = normalize_seed_url(u.strip())
            if nu not in seen_seed:
                seen_seed.add(nu)
                seeds.append(nu)
        if not seeds:
            logger.warning("No seed URLs for bank=%s", bank.bank_key)
            continue

        root_domain = urlparse(seeds[0]).netloc
        q: deque[tuple[str, int, str]] = deque((s, 0, "seed") for s in seeds)
        visited: set[str] = set()
        rows: list[dict] = []

        while q and len(visited) < max_pages:
            url, depth, parent = q.popleft()
            parsed = urlparse(url)
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            if normalized in visited:
                continue
            if parsed.netloc != root_domain:
                continue
            visited.add(normalized)

            try:
                result = fetcher.fetch(normalized)
            except Exception as exc:
                rows.append(
                    {
                        "url": normalized,
                        "depth": depth,
                        "parent_url": parent,
                        "http_status": "error",
                        "error": str(exc),
                        "inferred_topics": [],
                    }
                )
                continue

            soup = BeautifulSoup(result.html, "lxml")
            title = (soup.title.get_text(strip=True) if soup.title else "")[:200]
            topics = _classify_topics(normalized, title)
            rows.append(
                {
                    "url": normalized,
                    "final_url": result.final_url,
                    "depth": depth,
                    "parent_url": parent,
                    "http_status": result.status_code,
                    "page_title": title,
                    "inferred_topics": topics,
                    "review_recommended": bool(topics),
                }
            )

            if depth >= max_depth:
                continue

            for child in extract_same_domain_links(result.html, normalized):
                child = normalize_seed_url(child)
                if urlparse(child).netloc != root_domain:
                    continue
                if child in visited:
                    continue
                q.append((child, depth + 1, normalized))

        out_path = discovery_dir / f"{bank.bank_key}_url_candidates.jsonl"
        _write_jsonl(out_path, rows)
        logger.info(
            "Discovery finished bank=%s pages=%d output=%s",
            bank.bank_key,
            len(rows),
            out_path,
        )
