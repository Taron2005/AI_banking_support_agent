from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class BankMatch:
    bank_key: str
    matched_alias: str


_TOKEN_RE = re.compile(r"[\w\u0561-\u0587\u0531-\u0556]+", re.UNICODE)
# Short aliases: only use substring / fuzzy matching, not whitespace-stripped compact match
# (avoids false positives like "id" inside unrelated words when compacting).
_COMPACT_MATCH_MIN_ALIAS_LEN = 5


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    return t.casefold()


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _tokens(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN_RE.findall(_norm(text))]


class BankDetector:
    """Detect bank mentions in Armenian/English query text (exact + light fuzzy for typos)."""

    def __init__(self, aliases: dict[str, list[str]] | None = None) -> None:
        self._aliases = aliases or {
            "ameriabank": [
                "ameriabank",
                "ameria bank",
                "ameria",
                "ամերիա",
                "ամերիաբանկ",
                "ամերիա բանկ",
                "ամերիայի",
                "ամերիաբանկի",
            ],
            "acba": ["acba", "acba bank", "ակբա", "ակբա բանկ"],
            "idbank": ["idbank", "id bank", "այդիբանկ", "իդբանկ", "այդի բանկ"],
        }

    def _fuzzy_alias_hit(self, lower: str, alias: str, *, ratio: float = 0.84) -> bool:
        al = _norm(alias).strip()
        if len(al) < 4:
            return False
        if al in lower:
            return True
        max_delta = max(2, len(al) // 4)
        for tok in _tokens(lower):
            if abs(len(tok) - len(al)) > max_delta:
                continue
            if len(tok) < 4 and len(al) >= 6:
                continue
            if difflib.SequenceMatcher(None, tok, al).ratio() >= ratio:
                return True
        return False

    def _alias_hit_position(self, lower: str, compact_q: str, alias: str) -> int | None:
        """Return start index in *lower* space, or None. Uses NFC+casefold + optional compact match."""
        al = _norm(alias)
        if len(al) < 2:
            return None
        pos = lower.find(al)
        if pos != -1:
            return pos
        al_c = _compact(al)
        if len(al_c) >= _COMPACT_MATCH_MIN_ALIAS_LEN:
            cpos = compact_q.find(al_c)
            if cpos != -1:
                return cpos
        return None

    def detect_all(self, text: str) -> list[BankMatch]:
        """
        Every bank that matches at least one alias, ordered by first occurrence in the query.

        Exact substring matches win; then one fuzzy hit per bank (typos / STT glitches).
        """

        lower = _norm(text)
        compact_q = _compact(lower)
        per_bank: list[tuple[int, BankMatch]] = []
        fuzzy_added: set[str] = set()

        for bank_key, aliases in self._aliases.items():
            best_pos: int | None = None
            best_alias: str | None = None
            for alias in aliases:
                pos = self._alias_hit_position(lower, compact_q, alias)
                if pos is None:
                    continue
                if best_pos is None or pos < best_pos:
                    best_pos = pos
                    best_alias = alias
            if best_pos is not None and best_alias is not None:
                per_bank.append((best_pos, BankMatch(bank_key=bank_key, matched_alias=best_alias)))
                fuzzy_added.add(bank_key)

        for bank_key, aliases in self._aliases.items():
            if bank_key in fuzzy_added:
                continue
            for alias in aliases:
                if len(_norm(alias).strip()) < 4:
                    continue
                if self._fuzzy_alias_hit(lower, alias):
                    per_bank.append((len(lower), BankMatch(bank_key=bank_key, matched_alias=alias)))
                    fuzzy_added.add(bank_key)
                    break

        per_bank.sort(key=lambda x: x[0])
        return [m for _, m in per_bank]

    def detect(self, text: str) -> BankMatch | None:
        """Single best match: longest matched alias wins (backward compatible)."""

        all_m = self.detect_all(text)
        if not all_m:
            return None
        lower = _norm(text)
        best: BankMatch | None = None
        best_len = 0
        for m in all_m:
            al = _norm(m.matched_alias)
            if al in lower and len(al) > best_len:
                best = m
                best_len = len(al)
        if best is not None:
            return best
        return all_m[0]
