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
class AcbaScraper:
    """ACBA-specific scraping hints.

    The actual fetching and generic parsing is implemented in shared modules.
    This file exists to keep bank-specific heuristics/keywords localized.
    """

    bank_key: str = "acba"
    bank_name: str = "ACBA Bank"

    def fetch_structured(
        self,
        *,
        fetcher: RequestsHTMLFetcher,
        url: str,
        html: str,
        topic: TopicLabel,
    ) -> StructuredFetchResult:
        """
        API/JSON-first structured extractor.

        ACBA currently exposes limited stable public JSON endpoints for these pages,
        so this method prioritizes embedded JSON and then returns DOM fallback signal.
        """

        del fetcher  # API probing points can be added later without changing signature.
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
        ACBA-specific extraction rules.

        Why these exist:
        - ACBA pages contain repeated navigation/call widgets that pollute chunks.
        - Branch pages contain long lists where heading-based section parsing is weak.
        """

        return BankExtractionRules(
            remove_selectors=[
                SelectorRule(
                    selector="[class*='breadcrumb'], [id*='breadcrumb']",
                    why="Breadcrumb trails are navigation noise, not factual banking content.",
                ),
                SelectorRule(
                    selector="[class*='menu'], [class*='navbar'], [class*='nav-']",
                    why="Main menus create high-frequency noise terms that hurt retrieval precision.",
                ),
                SelectorRule(
                    selector="[class*='cookie'], [id*='cookie']",
                    why="Cookie banners are legally required UI but irrelevant for banking Q&A.",
                ),
                SelectorRule(
                    selector="[class*='feedback'], [id*='feedback']",
                    why="Feedback widgets add CTA phrases and duplicate contact text.",
                ),
                SelectorRule(
                    selector="[class*='call'], [class*='request']",
                    why="Call/request widgets are interaction UI and not product/branch facts.",
                ),
                SelectorRule(
                    selector="[class*='social'], [class*='share']",
                    why="Social share blocks add non-domain tokens that dilute embedding quality.",
                ),
            ],
            prefer_content_selectors=[
                SelectorRule(
                    selector="main, [role='main']",
                    why="Main content region typically excludes persistent header/footer blocks.",
                ),
                SelectorRule(
                    selector="[class*='content'], [class*='page-content'], [class*='inner']",
                    why="ACBA pages commonly keep article/product text inside content wrappers.",
                ),
            ],
            fallback_block_selectors=[
                SelectorRule(
                    selector="[class*='branch'], .branch, li, tr",
                    why="Branch pages often render repeated branch cards/list/table rows without headings.",
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
            phone_keywords=["հեռ", "phone", "телефон", "тел."],
        )

