from __future__ import annotations

"""
Shared phrases and helpers for “all banks” / broad questions vs bank-scoped retrieval.

Used by FollowUpResolver and RuntimeOrchestrator so wording stays in one place.
"""

# Phrases implying the user wants every bank in scope, not session.last_bank.
ALL_BANKS_QUERY_PHRASES: tuple[str, ...] = (
    # Armenian — explicit all / which banks
    "բոլոր բանկ",
    "բոլորի",
    "բոլոր բանկեր",
    "բոլորի մոտ",
    "բոլորում",
    "ինչ բանկեր",
    "որ բանկեր",
    "որ բանկերում",
    "որ բանկերից",
    "երեք բանկ",
    "մեր բանկեր",
    "այս բանկերը",
    "բանկերի մասին",
    "բանկերը թե",
    "բանկերը որ",
    "բանկերից որ",
    "բանկերում ինչ",
    "բանկերում որ",
    "ընդհանուր",
    "ընդհանուրապես",
    "զանգվածային",
    "համեմատ",
    "համեմատել",
    "համեմատությամբ",
    "դեմ համեմատ",
    "ամեն բանկ",
    "ամեն մի բանկ",
    # Switching away from one bank toward others (follow-ups)
    "ուրիշ բանկ",
    "այլ բանկ",
    "մյուս բանկ",
    "մնացած բանկ",
    "այլընտրանք",
    "նաև մյուս",
    "ոչ միայն",
    # English
    "each bank",
    "every bank",
    "all banks",
    "any bank",
    "which banks",
    "compare",
    "comparison",
    "between banks",
    "across banks",
    "versus",
    " vs ",
    " vs.",
)


# Explicit comparative intent (subset of phrases; does not require "all banks" wording).
COMPARISON_QUERY_PHRASES: tuple[str, ...] = (
    "համեմատ",
    "համեմատել",
    "համեմատությամբ",
    "դեմ համեմատ",
    "ով է ավելի",
    "որը ավելի",
    "compare",
    "comparison",
    "compared",
    "versus",
    " vs ",
    " vs.",
    " v.s.",
    "which is better",
    "which bank has",
    "who has the best",
    "higher rate",
    "lower rate",
    "ավելի բարձր",
    "ավելի ցածր",
)


def query_implies_comparison(lower: str) -> bool:
    return any(p in lower for p in COMPARISON_QUERY_PHRASES)


def query_implies_all_banks(lower: str) -> bool:
    return any(p in lower for p in ALL_BANKS_QUERY_PHRASES)


def should_diversify_across_banks(bank_keys: frozenset[str] | None) -> bool:
    """When retrieval should spread hits across banks (no single-bank pin)."""

    if bank_keys is None:
        return True
    return len(bank_keys) > 1
