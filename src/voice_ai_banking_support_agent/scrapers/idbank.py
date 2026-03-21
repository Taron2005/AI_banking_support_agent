from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from ..extraction.branch_parser import BranchParsingHints
from ..models import BranchRecord, TopicLabel
from .base import (
    BankExtractionRules,
    RequestsHTMLFetcher,
    SelectorRule,
    StructuredFetchResult,
    extract_same_domain_links,
    parse_inline_json_objects,
    parse_json_ld_objects,
)


@dataclass(frozen=True)
class IDBankScraper:
    """IDBank-specific scraping hints."""

    bank_key: str = "idbank"
    bank_name: str = "IDBank"

    def fetch_structured(
        self,
        *,
        fetcher: RequestsHTMLFetcher,
        url: str,
        html: str,
        topic: TopicLabel,
    ) -> StructuredFetchResult:
        """
        API/JSON-first structured extractor for IDBank pages.

        We first inspect JSON-LD and inline state blocks, then pass control to
        shared DOM branch parser in pipeline when no structured records are found.
        """

        del fetcher
        notes: list[str] = []
        branch_records: list[BranchRecord] = []

        json_objs: list[dict[str, Any]] = parse_json_ld_objects(html) + parse_inline_json_objects(html)
        if json_objs:
            notes.append(f"embedded_json_blocks={len(json_objs)}")

        discovered_urls = extract_same_domain_links(html, url)
        if topic == "branch":
            discovered_urls = [u for u in discovered_urls if "branch" in u.lower() or "atm" in u.lower()]
        return StructuredFetchResult(
            branch_records=branch_records,
            discovered_urls=discovered_urls,
            notes=notes,
            supplemental_html=None,
        )

    def extraction_rules(self) -> BankExtractionRules:
        """
        IDBank-specific extraction rules.

        Why these exist:
        - IDBank branch pages include app-download banners, poll widgets, and repeated CTA controls.
        - Branch content often appears in repeated blocks requiring fallback block parsing.
        """

        return BankExtractionRules(
            remove_selectors=[
                SelectorRule(
                    selector="[class*='breadcrumb'], [id*='breadcrumb']",
                    why="Breadcrumbs add path tokens and dilute branch/address retrieval.",
                ),
                SelectorRule(
                    selector="[class*='menu'], [class*='navbar']",
                    why="Top navigation labels are high-frequency noise across pages.",
                ),
                SelectorRule(
                    selector="[class*='cookie'], [id*='cookie']",
                    why="Cookie/legal controls are not relevant to banking knowledge chunks.",
                ),
                SelectorRule(
                    selector="[class*='download'], [class*='idram']",
                    why="App download promo banners are unrelated to branch/credit/deposit facts.",
                ),
                SelectorRule(
                    selector="[class*='poll'], [class*='survey'], [class*='question']",
                    why="Embedded feedback/poll widgets inject unrelated multi-language text.",
                ),
                SelectorRule(
                    selector="[class*='more'], [class*='hide']",
                    why="Show-more/hide controls are purely UI and repeatedly duplicated.",
                ),
            ],
            prefer_content_selectors=[
                SelectorRule(
                    selector="main, [role='main']",
                    why="Primary content region has highest chance of containing structured branch data.",
                ),
                SelectorRule(
                    selector="[class*='branch'], [class*='content'], [class*='page-content']",
                    why="IDBank branch and product data is often nested in branch/content wrappers.",
                ),
            ],
            fallback_block_selectors=[
                SelectorRule(
                    selector="[class*='branch'], [class*='item'], li, tr",
                    why="Branch pages are often repeated branch blocks rather than heading sections.",
                )
            ],
        )

    def branch_parsing_hints(self) -> BranchParsingHints:
        return BranchParsingHints(
            branch_name_keywords=["մասնաճյուղ", "branch", "filial", "филиал"],
            city_keywords=["քաղաք", "city", "город"],
            district_keywords=["թաղամաս", "district", "район"],
            address_keywords=["հասցե", "address", "адрес"],
            working_hours_keywords=["աշխ", "աշխ. ժամ", "աշխատ", "hours", "working hours", "график"],
            phone_keywords=["հեռ", "phone", "телефон", "тел.", "+374"],
        )

