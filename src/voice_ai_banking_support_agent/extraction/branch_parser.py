from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup

from ..models import BranchRecord
from .cleaning import normalize_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BranchParsingHints:
    """Keyword hints for mapping columns when parsing branch pages."""

    branch_name_keywords: list[str]
    city_keywords: list[str]
    district_keywords: list[str]
    address_keywords: list[str]
    working_hours_keywords: list[str]
    phone_keywords: list[str]


PHONE_RE = re.compile(r"(\+?374[\s-]?\d{2,3}[\s-]?\d{2,3}[\s-]?\d{2,3})")


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _extract_phone(text: str) -> str | None:
    m = PHONE_RE.search(text)
    if not m:
        return None
    return normalize_text(m.group(1))


def _extract_hours(text: str) -> str | None:
    # Very rough: hours often include `:` times.
    if re.search(r"\b\d{1,2}:\d{2}\b", text):
        return normalize_text(text)
    # Also detect common weekday abbreviations/words.
    if any(w in text.lower() for w in ["mon", "tue", "wed", "thu", "fri", "sat", "sun", "երկ", "երք", "չրք", "ուրբ", "հնգ", "շբթ", "կիր"]):
        return normalize_text(text)
    return None


def _map_table_columns(headers: list[str], hints: BranchParsingHints) -> dict[str, int]:
    """
    Map semantic fields to column indices based on header keywords.

    Returns mapping keys:
      - branch_name, city, district, address, working_hours, phone
    """

    mapped: dict[str, int] = {}
    for idx, header in enumerate(headers):
        header_norm = header.strip().lower()
        if not header_norm:
            continue

        if _contains_any_keyword(header, hints.branch_name_keywords):
            mapped.setdefault("branch_name", idx)
        if _contains_any_keyword(header, hints.city_keywords):
            mapped.setdefault("city", idx)
        if _contains_any_keyword(header, hints.district_keywords):
            mapped.setdefault("district", idx)
        if _contains_any_keyword(header, hints.address_keywords):
            mapped.setdefault("address", idx)
        if _contains_any_keyword(header, hints.working_hours_keywords):
            mapped.setdefault("working_hours", idx)
        if _contains_any_keyword(header, hints.phone_keywords):
            mapped.setdefault("phone", idx)

    return mapped


def parse_branch_records(
    html: str,
    *,
    bank_name: str,
    source_url: str,
    cleaned_text: str | None = None,
    hints: BranchParsingHints,
) -> list[BranchRecord]:
    """
    Parse a branch page and return structured branch records (best-effort).

    Args:
        html: Raw HTML of the branch page.
        bank_name: Bank display name (used in output records).
        source_url: The source page URL.
        cleaned_text: Optional cleaned page text for fallback parsing.
        hints: Bank-specific keyword hints.

    Returns:
        A list of `BranchRecord` objects.
    """

    soup = BeautifulSoup(html, "lxml")

    # Primary strategy: parse HTML tables where possible.
    tables = soup.find_all("table")
    records: list[BranchRecord] = []

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        # Extract header texts from the first row containing <th>.
        header_cells: list[str] = []
        header_row = None
        for r in rows[:5]:
            ths = r.find_all("th")
            if ths:
                header_row = r
                header_cells = [th.get_text(separator=" ", strip=True) for th in ths]
                break
        if not header_cells or header_row is None:
            continue

        column_map = _map_table_columns(header_cells, hints)
        # Require address OR (city + branch_name) for minimum usefulness.
        if "address" not in column_map and "city" not in column_map:
            continue

        data_rows = table.find_all("tr")[len(table.find_all("tr")) - len(rows) :]  # no-op safety
        # Better: consider rows after the header_row.
        header_index = rows.index(header_row)
        data_rows = rows[header_index + 1 :]

        for tr in data_rows:
            tds = tr.find_all("td")
            if not tds:
                continue

            cells = [td.get_text(separator=" ", strip=True) for td in tds]
            if not cells:
                continue

            def cell_or_none(field: str) -> str | None:
                idx = column_map.get(field)
                if idx is None:
                    return None
                if idx < 0 or idx >= len(cells):
                    return None
                return normalize_text(cells[idx])

            branch_name = cell_or_none("branch_name")
            city = cell_or_none("city")
            district = cell_or_none("district")
            address = cell_or_none("address")
            working_hours = cell_or_none("working_hours")

            phone = cell_or_none("phone")

            # Additional enrichment from any cell if phone/hours missing.
            if phone is None:
                for c in cells:
                    ph = _extract_phone(c)
                    if ph:
                        phone = ph
                        break
            if working_hours is None:
                for c in cells:
                    hrs = _extract_hours(c)
                    if hrs:
                        working_hours = hrs
                        break

            # If we still don't have a branch_name, use the first non-empty cell.
            if not branch_name:
                for c in cells:
                    if c and len(c) > 3:
                        branch_name = normalize_text(c)
                        break

            if not city:
                # Sometimes city is encoded in address. Keep best-effort.
                if address:
                    city = ""

            if not branch_name or not (address or city):
                continue

            if not address:
                logger.warning("Branch record missing address for %s at %s", bank_name, source_url)
                # Keep it anyway; validation will decide whether to drop later.
                address = ""

            records.append(
                BranchRecord(
                    bank_name=bank_name,
                    branch_name=branch_name,
                    city=city or "",
                    district=district,
                    address=address,
                    working_hours=working_hours,
                    phone=phone,
                    source_url=source_url,
                )
            )

    if records:
        # De-duplicate exact matches.
        unique: dict[tuple[str, str, str], BranchRecord] = {}
        for r in records:
            key = (r.branch_name, r.city, r.address)
            unique[key] = r
        return list(unique.values())

    # Fallback strategy: parse from cleaned text lines using regex.
    if cleaned_text:
        return _parse_branch_records_from_text_lines(
            cleaned_text=cleaned_text,
            bank_name=bank_name,
            source_url=source_url,
            hints=hints,
        )

    logger.warning("No structured branch records found (no tables and no cleaned fallback).")
    return []


def _parse_branch_records_from_text_lines(
    *,
    cleaned_text: str,
    bank_name: str,
    source_url: str,
    hints: BranchParsingHints,
) -> list[BranchRecord]:
    """
    Best-effort fallback parsing from text.

    This is less reliable than table parsing, but helps when branch pages
    don't use well-structured tables.
    """

    lines = [ln.strip() for ln in cleaned_text.split("\n") if ln.strip()]

    records: list[BranchRecord] = []
    current: dict[str, str] = {}

    def flush() -> None:
        nonlocal current
        if not current:
            return
        branch_name = current.get("branch_name", "").strip()
        city = current.get("city", "").strip()
        address = current.get("address", "").strip()
        if branch_name or address or city:
            records.append(
                BranchRecord(
                    bank_name=bank_name,
                    branch_name=branch_name or "Unknown branch",
                    city=city,
                    district=current.get("district"),
                    address=address,
                    working_hours=current.get("working_hours"),
                    phone=current.get("phone"),
                    source_url=source_url,
                )
            )
        current = {}

    for ln in lines:
        # Start a new record if the line strongly looks like a branch heading.
        if _contains_any_keyword(ln, hints.branch_name_keywords) and len(ln) > 4:
            flush()
            current["branch_name"] = ln
            continue

        # Map fields by keyword presence.
        if _contains_any_keyword(ln, hints.city_keywords):
            current["city"] = ln
            continue
        if _contains_any_keyword(ln, hints.district_keywords):
            current["district"] = ln
            continue
        if _contains_any_keyword(ln, hints.address_keywords):
            current["address"] = ln
            continue

        if not current.get("phone"):
            ph = _extract_phone(ln)
            if ph:
                current["phone"] = ph
                continue

        if not current.get("working_hours"):
            hrs = _extract_hours(ln)
            if hrs:
                current["working_hours"] = hrs
                continue

        # If we already started a record but don't know address, allow the next line
        # to become address if it looks address-like (digits + street).
        if not current.get("address") and re.search(r"\b\d{1,4}\b", ln):
            current["address"] = ln

    flush()
    if not records:
        logger.warning("Branch fallback parsing found 0 records for %s.", source_url)
    return records

