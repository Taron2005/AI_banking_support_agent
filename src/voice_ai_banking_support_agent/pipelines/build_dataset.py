from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from ..bank_manifest import load_banks_manifest, manifest_summary
from ..config import AppConfig
from ..extraction.branch_parser import parse_branch_records
from ..extraction.cleaning import clean_html_to_text, detect_language_from_text
from ..extraction.section_parser import parse_sections_from_html
from ..indexing.chunker import chunk_sections
from ..models import TopicLabel
from ..scrapers.acba import AcbaScraper
from ..scrapers.ameriabank import AmeriaBankScraper
from ..scrapers.idbank import IDBankScraper
from ..scrapers.base import RequestsHTMLFetcher, extract_page_title
from ..utils.text import stable_id
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


BANK_SCRAPERS = {
    "acba": AcbaScraper,
    "ameriabank": AmeriaBankScraper,
    "idbank": IDBankScraper,
}


def _topic_to_attr(topic: TopicLabel) -> str:
    if topic == "credit":
        return "credits"
    if topic == "deposit":
        return "deposits"
    if topic == "branch":
        return "branches"
    raise ValueError(f"Unknown topic: {topic}")


class _DedupJsonlAppender:
    """
    Append JSONL rows with per-file in-memory dedupe caches.

    This avoids re-reading large JSONL files for every page while preserving
    deterministic de-duplication behavior.
    """

    def __init__(self) -> None:
        self._keys_by_file: dict[tuple[Path, tuple[str, ...]], set[tuple[str, ...]]] = {}

    def append(self, path: Path, records: Iterable[dict], *, unique_key_fields: list[str]) -> int:
        key_spec = tuple(unique_key_fields)
        cache_key = (path, key_spec)
        existing_keys = self._keys_by_file.get(cache_key)
        if existing_keys is None:
            existing_keys = set()
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        row_key = tuple(str(obj.get(k, "")) for k in unique_key_fields)
                        existing_keys.add(row_key)
            self._keys_by_file[cache_key] = existing_keys

        new_rows: list[dict] = []
        for row in records:
            row_key = tuple(str(row.get(k, "")) for k in unique_key_fields)
            if row_key in existing_keys:
                continue
            existing_keys.add(row_key)
            new_rows.append(row)

        if not new_rows:
            return 0

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(new_rows)


def _safe_for_log(text: str) -> str:
    """Prevent console encoding errors when logs contain Armenian text."""
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def build_dataset(
    *,
    config: AppConfig,
    banks: list[str] | None,
    topics: list[TopicLabel],
) -> None:
    """
    Build offline dataset artifacts for ingestion + indexing.

    Outputs:
        - raw HTML artifacts
        - cleaned page JSONL (page-level)
        - branches JSONL (structured records, topic=branch)
        - chunked documents JSONL (embedding input)
    """

    config.ensure_dirs()
    setup_logging()

    manifest = load_banks_manifest(config.manifest_path)
    logger.info("Loaded manifest:\n%s", manifest_summary(manifest))

    fetcher = RequestsHTMLFetcher(config.network)
    writer = _DedupJsonlAppender()
    processed = 0
    failed = 0

    selected_banks = {b.lower() for b in (banks or [])} if banks else None
    for bank in manifest.banks:
        if selected_banks is not None and bank.bank_key.lower() not in selected_banks:
            continue

        bank_key = bank.bank_key
        bank_name = bank.bank_name
        scraper_cls = BANK_SCRAPERS.get(bank_key.lower())
        if scraper_cls is None:
            logger.warning("No scraper implementation for bank_key=%s. Skipping.", bank_key)
            continue
        scraper = scraper_cls()
        hints = scraper.branch_parsing_hints()
        extraction_rules = scraper.extraction_rules()

        for topic in topics:
            attr = _topic_to_attr(topic)
            topic_obj = getattr(bank, attr)
            urls = topic_obj.urls
            if not urls:
                logger.warning("No URLs configured for %s topic=%s", bank_key, topic)
                continue

            logger.info("Processing bank=%s topic=%s url_count=%d", bank_key, topic, len(urls))

            # Output paths
            chunks_path = config.chunks_dir / f"{bank_key}_{topic}_chunks.jsonl"
            cleaned_pages_path = config.cleaned_docs_dir / f"{bank_key}_{topic}_pages.jsonl"
            branches_path = config.branches_dir / f"{bank_key}_branches.jsonl"
            if topic == "branch":
                # Always create output file for predictable downstream workflows.
                branches_path.parent.mkdir(parents=True, exist_ok=True)
                branches_path.touch(exist_ok=True)

            for url in urls:
                # Save raw HTML as an artifact for inspection/debugging.
                page_id = stable_id(bank_key, topic, url)
                raw_path = config.raw_html_dir / bank_key / topic / f"{page_id}.html"

                try:
                    if url.lower().endswith(".pdf"):
                        logger.warning(
                            "Skipping non-HTML source bank=%s topic=%s url=%s. "
                            "Current pipeline handles HTML pages only.",
                            bank_key,
                            topic,
                            url,
                        )
                        continue
                    if config.reuse_raw_html and raw_path.exists():
                        html = raw_path.read_text(encoding="utf-8", errors="ignore")
                        logger.info("Reusing cached HTML: %s", raw_path)
                    else:
                        result = fetcher.fetch(url)
                        html = result.html
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_path.write_text(html, encoding="utf-8")

                    page_title = extract_page_title(html) or f"{bank_name} {topic}"
                    structured = scraper.fetch_structured(
                        fetcher=fetcher,
                        url=url,
                        html=html,
                        topic=topic,
                    )
                    for note in structured.notes:
                        logger.info("Structured extractor note bank=%s url=%s note=%s", bank_key, url, note)

                    cleaning = clean_html_to_text(html, rules=extraction_rules)
                    supplemental_html = structured.supplemental_html or ""
                    supplemental_cleaning = (
                        clean_html_to_text(supplemental_html, rules=None) if supplemental_html else None
                    )
                    if (
                        not cleaning.usable
                        and supplemental_cleaning is not None
                        and supplemental_cleaning.usable
                    ):
                        # For API-loaded pages (e.g., DNN modules), use extracted payload HTML.
                        cleaning = supplemental_cleaning
                        logger.info(
                            "Using structured supplemental payload for content bank=%s topic=%s url=%s",
                            bank_key,
                            topic,
                            url,
                        )
                    if not cleaning.usable:
                        # Bank pages can be compact card layouts; try section fallback before skipping.
                        candidate_sections = parse_sections_from_html(
                            html,
                            max_sections=config.chunking.max_sections_per_page,
                            min_content_chars=40,
                            rules=extraction_rules,
                        )
                        if not candidate_sections:
                            logger.warning(
                                "Skipping low-value extraction bank=%s topic=%s url=%s warning=%s",
                                bank_key,
                                topic,
                                url,
                                cleaning.warning,
                            )
                            continue

                    detected_language = detect_language_from_text(cleaning.cleaned_text)
                    if bank.language == "hy" and detected_language != "hy":
                        logger.warning(
                            "Detected non-Armenian dominant text for bank=%s topic=%s url=%s. "
                            "Check manifest URL language path.",
                            bank_key,
                            topic,
                            url,
                        )

                    # Always save cleaned page content.
                    writer.append(
                        cleaned_pages_path,
                        [
                            {
                                "bank_key": bank_key,
                                "bank_name": bank_name,
                                "topic": topic,
                                "source_url": url,
                                "page_title": page_title,
                                "language": detected_language,
                                "raw_text": cleaning.raw_text,
                                "cleaned_text": cleaning.cleaned_text,
                            }
                        ],
                        unique_key_fields=["bank_key", "topic", "source_url"],
                    )

                    if topic == "branch":
                        branch_records = structured.branch_records or parse_branch_records(
                            html + ("\n" + supplemental_html if supplemental_html else ""),
                            bank_name=bank_name,
                            source_url=url,
                            cleaned_text=cleaning.cleaned_text,
                            hints=hints,
                        )
                        if not branch_records:
                            logger.warning(
                                "Branch parser produced 0 records bank=%s url=%s", bank_key, url
                            )
                        else:
                            writer.append(
                                branches_path,
                                [
                                    {
                                        **r.model_dump(),
                                        "bank_key": bank_key,
                                    }
                                    for r in branch_records
                                ],
                                unique_key_fields=["bank_key", "source_url", "branch_name", "address"],
                            )

                    sections = parse_sections_from_html(
                        html,
                        max_sections=config.chunking.max_sections_per_page,
                        rules=extraction_rules,
                    )
                    if not sections and supplemental_html:
                        sections = parse_sections_from_html(
                            supplemental_html,
                            max_sections=config.chunking.max_sections_per_page,
                            rules=None,
                        )
                    if not sections:
                        logger.warning("No sections found bank=%s topic=%s url=%s", bank_key, topic, url)
                        continue

                    docs = chunk_sections(
                        sections=sections,
                        bank_key=bank_key,
                        bank_name=bank_name,
                        topic=topic,
                        source_url=url,
                        page_title=page_title,
                        language=detected_language,
                        raw_page_text=cleaning.raw_text,
                        chunking=config.chunking,
                    )

                    if not docs:
                        logger.warning(
                            "Chunker produced 0 documents bank=%s topic=%s url=%s page_title=%s",
                            bank_key,
                            topic,
                            url,
                            _safe_for_log(page_title),
                        )
                        continue

                    writer.append(
                        chunks_path,
                        [d.model_dump() for d in docs],
                        unique_key_fields=["chunk_id"],
                    )
                    processed += 1

                except Exception as e:
                    logger.exception(
                        "Failed processing bank=%s topic=%s url=%s. Error=%s",
                        bank_key,
                        topic,
                        url,
                        str(e),
                    )
                    failed += 1
                    continue

    total = processed + failed
    if total > 0:
        fail_ratio = failed / total
        logger.info("Dataset build finished: processed=%d failed=%d fail_ratio=%.2f", processed, failed, fail_ratio)
        if fail_ratio > 0.5:
            raise RuntimeError(
                f"Too many page failures during dataset build ({failed}/{total}). "
                "Check logs and manifest URLs before proceeding."
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline dataset builder")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    p.add_argument("--project-root", type=str, default=".", help="Repository root path.")
    p.add_argument("--banks", nargs="*", default=None, help="Bank keys to process, e.g. acba idbank.")
    p.add_argument("--topics", nargs="+", default=["credit", "deposit", "branch"], help="Topics to build.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    setup_logging(args.log_level)

    project_root = Path(args.project_root).resolve()
    config_yaml = Path(args.config).resolve() if args.config else None
    cfg = AppConfig(
        project_root=project_root,
        manifest_path=project_root / "manifests" / "banks.yaml",
        data_dir=project_root / "data",
        raw_html_dir=project_root / "data" / "raw_html",
        branches_dir=project_root / "data" / "branches",
        cleaned_docs_dir=project_root / "data" / "cleaned_docs",
        chunks_dir=project_root / "data" / "chunks",
        index_dir=project_root / "data" / "index",
        embedding_model_name="Metric-AI/armenian-text-embeddings-2-large",
    )

    if config_yaml:
        # Use the generic loader so nested settings apply.
        from ..config import load_config

        cfg = load_config(project_root=project_root, config_yaml=config_yaml)

    topics: list[TopicLabel] = [t.lower() for t in args.topics]  # type: ignore[list-item]
    build_dataset(config=cfg, banks=args.banks, topics=topics)


if __name__ == "__main__":
    main()

