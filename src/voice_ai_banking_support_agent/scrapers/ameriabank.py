from __future__ import annotations

import re
from typing import Any
from dataclasses import dataclass
from urllib.parse import urlparse

from bs4 import BeautifulSoup
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
class AmeriaBankScraper:
    """Ameriabank-specific scraping hints."""

    bank_key: str = "ameriabank"
    bank_name: str = "Ameriabank"

    def fetch_structured(
        self,
        *,
        fetcher: RequestsHTMLFetcher,
        url: str,
        html: str,
        topic: TopicLabel,
    ) -> StructuredFetchResult:
        """
        API/JSON-first structured extractor with Ameriabank-specific branch fallback.

        Order:
        1) direct API endpoint probing (service-network related)
        2) JSON-LD / inline JS data blocks
        3) DOM fallback specialized for service-network if still empty
        """

        notes: list[str] = []
        branch_records: list[BranchRecord] = []
        discovered_urls = extract_same_domain_links(html, url)
        supplemental_html = self._fetch_dnn_module_html(fetcher=fetcher, url=url, html=html, notes=notes)

        if topic != "branch":
            return StructuredFetchResult(
                branch_records=[],
                discovered_urls=discovered_urls,
                notes=notes,
                supplemental_html=supplemental_html,
            )

        api_records = self._try_service_network_api(fetcher=fetcher, url=url, html=html, notes=notes)
        if api_records:
            branch_records.extend(api_records)
            notes.append(f"api_records={len(api_records)}")

        if not branch_records:
            json_records = self._extract_branch_records_from_embedded_json(url=url, html=html, notes=notes)
            if json_records:
                branch_records.extend(json_records)
                notes.append(f"embedded_json_records={len(json_records)}")

        if not branch_records:
            dom_html = html
            if supplemental_html:
                dom_html = html + "\n" + supplemental_html
            dom_records = self._extract_branch_records_from_dom(url=url, html=dom_html)
            if dom_records:
                branch_records.extend(dom_records)
                notes.append(f"dom_records={len(dom_records)}")

        branch_links = [u for u in discovered_urls if "service-network" in u.lower() or "branch" in u.lower()]
        return StructuredFetchResult(
            branch_records=self._dedupe_records(branch_records),
            discovered_urls=branch_links or discovered_urls,
            notes=notes,
            supplemental_html=supplemental_html,
        )

    def _fetch_dnn_module_html(
        self,
        *,
        fetcher: RequestsHTMLFetcher,
        url: str,
        html: str,
        notes: list[str],
    ) -> str | None:
        """
        Fetch module payload HTML via DNN API for pages with empty containers.

        Ameriabank pages often include `initModule(...)` stubs with `hasViewContent=false`,
        meaning useful page content is loaded by client-side API calls.
        """

        module_ids = list(dict.fromkeys(re.findall(r"initModule\((\d+),\s*args\)", html)))
        if not module_ids:
            return None

        tab_match = re.search(r"tabId:\s*(\d+)", html)
        tab_id = int(tab_match.group(1)) if tab_match else 0
        token_match = re.search(
            r'name="__RequestVerificationToken"\s+type="hidden"\s+value="([^"]+)"',
            html,
        )
        token = token_match.group(1) if token_match else ""

        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        site_root = "/en/" if path_lower.startswith("/en/") else "/"
        endpoint = f"{parsed.scheme}://{parsed.netloc}{site_root}API/WebsitesCreative/MyContentManager/API/Init"

        fragments: list[str] = []
        for module_id in module_ids[:20]:
            headers = {
                "ModuleId": str(module_id),
                "TabId": str(tab_id),
                "RequestVerificationToken": token,
                "X-Requested-With": "XMLHttpRequest",
            }
            try:
                status, body = fetcher.fetch_json(
                    endpoint,
                    method="GET",
                    params={"portalId": 0, "tabId": tab_id, "moduleId": int(module_id)},
                    headers=headers,
                )
            except Exception:
                continue
            if status != 200 or not isinstance(body, dict):
                continue
            data = body.get("data")
            if not isinstance(data, dict):
                continue
            slides = data.get("slides")
            if not isinstance(slides, list):
                continue
            for slide in slides:
                if not isinstance(slide, dict):
                    continue
                desc = slide.get("description")
                if isinstance(desc, dict):
                    original = desc.get("original")
                    if isinstance(original, str) and len(original) > 40:
                        fragments.append(original)
                for val in slide.values():
                    if isinstance(val, str) and len(val) > 120 and "<" in val:
                        fragments.append(val)

        if not fragments:
            return None
        combined = "\n".join(fragments)
        notes.append(f"dnn_module_payload_fragments={len(fragments)}")
        return combined

    def _try_service_network_api(
        self,
        *,
        fetcher: RequestsHTMLFetcher,
        url: str,
        html: str,
        notes: list[str],
    ) -> list[BranchRecord]:
        """
        Probe likely DNN API endpoints used by service-network modules.

        This is best-effort and non-fatal: if endpoints are unavailable externally,
        we fall back to embedded JSON and DOM parsing.
        """

        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        module_ids = sorted(set(re.findall(r"initModule\((\d+),\s*args\)", html)))
        if not module_ids:
            return []

        candidates = [
            f"{base}/API/services/WebsitesCreative/MyContentManager/API/Init",
            f"{base}/API/services/WebSitesCreative/MyContentManager/API/Init",
            f"{base}/api/services/websitescreative/mycontentmanager/api/init",
            f"{base}/DesktopModules/WebSitesCreative/MyContentManager/API/Init",
        ]
        tab_match = re.search(r"tabId:\s*(\d+)", html)
        tab_id = int(tab_match.group(1)) if tab_match else 3420

        for module_id in module_ids[:20]:
            payload = {"portalId": 0, "tabId": tab_id, "moduleId": int(module_id)}
            for endpoint in candidates:
                status, body = fetcher.fetch_json(endpoint, method="POST", json_body=payload)
                if status >= 400 or body is None:
                    continue
                notes.append(f"api_hit={endpoint} module_id={module_id}")
                records = self._records_from_unknown_json(body, url)
                if records:
                    return records
        return []

    def _extract_branch_records_from_embedded_json(
        self,
        *,
        url: str,
        html: str,
        notes: list[str],
    ) -> list[BranchRecord]:
        json_objs: list[dict[str, Any]] = parse_json_ld_objects(html) + parse_inline_json_objects(html)
        if json_objs:
            notes.append(f"embedded_json_blocks={len(json_objs)}")
        records: list[BranchRecord] = []
        for obj in json_objs:
            records.extend(self._records_from_unknown_json(obj, url))
        return self._dedupe_records(records)

    def _records_from_unknown_json(self, obj: Any, source_url: str) -> list[BranchRecord]:
        records: list[BranchRecord] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                lowered = {str(k).lower(): v for k, v in node.items()}
                address = self._pick_str(lowered, ["address", "fulladdress", "location", "addr"])
                city = self._pick_str(lowered, ["city", "town"])
                name = self._pick_str(lowered, ["name", "title", "branchname", "label"])
                phone = self._pick_str(lowered, ["phone", "phone_number", "telephone", "tel"])
                hours = self._pick_str(lowered, ["workinghours", "hours", "schedule", "worktime"])
                if address and (name or city):
                    records.append(
                        BranchRecord(
                            bank_name=self.bank_name,
                            branch_name=name or "Unknown branch",
                            city=city or "",
                            address=address,
                            working_hours=hours,
                            phone=phone,
                            source_url=source_url,
                        )
                    )
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(obj)
        return self._dedupe_records(records)

    def _extract_branch_records_from_dom(self, *, url: str, html: str) -> list[BranchRecord]:
        """
        Last fallback for service-network when API/JSON data is unavailable.

        We intentionally keep this conservative to avoid false positives.
        """

        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        phone_re = re.compile(r"(\+?374[\s()/-]?\d{2,3}[\s()/-]?\d{2,3}[\s()/-]?\d{2,4})")
        addr_re = re.compile(r".*(Երևան|Yerevan).*\d+.*")
        candidates: list[tuple[str, str | None]] = []
        for i, line in enumerate(lines):
            if addr_re.match(line):
                near = " ".join(lines[max(0, i - 2) : min(len(lines), i + 3)])
                m = phone_re.search(near)
                candidates.append((line, m.group(1) if m else None))

        if not candidates:
            # Known service-network page footer often includes HQ address/phone.
            m_addr = re.search(r"(Վազգեն Սարգսյան\s*\d+[^,\n]*,\s*Երևան[^\n]*)", html)
            m_phone = re.search(r"(\+?374[\s()/-]?10[\s()/-]?56[\s()/-]?11[\s()/-]?11)", html)
            if m_addr:
                candidates = [(m_addr.group(1), m_phone.group(1) if m_phone else None)]

        out: list[BranchRecord] = []
        for addr, phone in candidates[:25]:
            out.append(
                BranchRecord(
                    bank_name=self.bank_name,
                    branch_name="Service Network Branch",
                    city="Երևան" if "Երևան" in addr else "",
                    address=addr,
                    working_hours=None,
                    phone=phone,
                    source_url=url,
                )
            )
        return self._dedupe_records(out)

    @staticmethod
    def _pick_str(obj: dict[str, Any], keys: list[str]) -> str | None:
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    @staticmethod
    def _dedupe_records(records: list[BranchRecord]) -> list[BranchRecord]:
        unique: dict[tuple[str, str, str], BranchRecord] = {}
        for r in records:
            key = (r.branch_name.strip(), r.city.strip(), r.address.strip())
            if key[2]:
                unique[key] = r
        return list(unique.values())

    def extraction_rules(self) -> BankExtractionRules:
        """
        Ameriabank-specific extraction rules.

        Why these exist:
        - Ameriabank pages rely on cards/sliders and many UI wrappers.
        - Product pages can be short in plain paragraphs, so fallback block parsing helps.
        """

        return BankExtractionRules(
            remove_selectors=[
                SelectorRule(
                    selector="[class*='breadcrumb'], [id*='breadcrumb']",
                    why="Breadcrumbs duplicate path labels and should not enter retrieval chunks.",
                ),
                SelectorRule(
                    selector="[class*='menu'], [class*='navbar'], [class*='header-nav']",
                    why="Navigation menus dominate token distribution with irrelevant labels.",
                ),
                SelectorRule(
                    selector="[class*='cookie'], [id*='cookie']",
                    why="Cookie controls are boilerplate and reduce chunk signal quality.",
                ),
                SelectorRule(
                    selector="[class*='modal'], [class*='popup'], [class*='dialog']",
                    why="Modal/popup blocks frequently contain transient marketing content.",
                ),
                SelectorRule(
                    selector="[class*='chat'], [class*='assistant']",
                    why="Chat widgets add noise and can leak repeated support CTA text.",
                ),
            ],
            prefer_content_selectors=[
                SelectorRule(
                    selector="main, [role='main']",
                    why="Main region is the most reliable area for product/deposit descriptions.",
                ),
                SelectorRule(
                    selector="[class*='product'], [class*='article'], [class*='service']",
                    why="Ameriabank often uses product/service cards rather than long prose.",
                ),
            ],
            fallback_block_selectors=[
                SelectorRule(
                    selector="[class*='card'], [class*='item'], [class*='service'], li",
                    why="Fallback from card/list layouts when heading-based parse yields few sections.",
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

