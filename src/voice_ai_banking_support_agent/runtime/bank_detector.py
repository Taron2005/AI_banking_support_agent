from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BankMatch:
    bank_key: str
    matched_alias: str


class BankDetector:
    """Detect bank mentions in Armenian/English query text."""

    def __init__(self, aliases: dict[str, list[str]] | None = None) -> None:
        self._aliases = aliases or {
            "ameriabank": ["ameriabank", "ameria", "ամերիա", "ամերիաբանկ"],
            "acba": ["acba", "ակբա", "acba bank"],
            "idbank": ["idbank", "id bank", "այդիբանկ", "իդբանկ"],
        }

    def detect(self, text: str) -> BankMatch | None:
        lower = text.lower()
        for bank_key, aliases in self._aliases.items():
            for alias in aliases:
                if alias in lower:
                    return BankMatch(bank_key=bank_key, matched_alias=alias)
        return None

