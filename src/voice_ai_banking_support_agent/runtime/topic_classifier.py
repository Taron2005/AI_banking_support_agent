from __future__ import annotations

import re
from dataclasses import dataclass

from .lexical_fuzzy import fuzzy_term_matches
from .models import TopicClassification


@dataclass(frozen=True)
class TopicClassifierConfig:
    ambiguous_margin: float = 0.15
    strong_term_weight: float = 1.5
    weak_term_weight: float = 0.75
    fuzzy_match: bool = True
    fuzzy_ratio: float = 0.8


class TopicClassifier:
    """
    Rules-first topic and request-type classifier with fuzzy token matching for STT/typos.

    Labels:
    - credit / deposit / branch
    - out_of_scope
    - ambiguous
    - unsupported_request_type
    """

    def __init__(self, cfg: TopicClassifierConfig | None = None) -> None:
        self._cfg = cfg or TopicClassifierConfig()

    _strong_topic_terms: dict[str, tuple[str, ...]] = {
        "credit": (
            "վարկ",
            "վարկեր",
            "վարկի",
            "վարկը",
            "սպառողական",
            "սպառողական վարկ",
            "ipoteca",
            "ipotek",
            "loan",
            "loans",
            "mortgage",
            "credit",
            "lending",
        ),
        "deposit": (
            "ավանդ",
            "ավանդներ",
            "ավանդի",
            "ավանդը",
            "խնայող",
            "խնայողական",
            "deposit",
            "deposits",
            "savings",
            "term deposit",
        ),
        "branch": (
            "մասնաճյուղ",
            "մասնաճյուղեր",
            "մասնաճյուղի",
            "բանկոմատ",
            "հասցե",
            "որտե՞ղ",
            "որտեղ",
            "where",
            "address",
            "branch",
            "atm",
            "location",
        ),
    }
    _weak_topic_terms: dict[str, tuple[str, ...]] = {
        "credit": (
            "interest rate",
            "տոկոս",
            "տոկոսադրույք",
            "տոկոսադրույքներ",
            "monthly",
            "անվանական",
            "սահմանաչափ",
        ),
        "deposit": (
            "interest",
            "rate",
            "dollar",
            "dram",
            "currency",
            "cumulative deposit",
            "տոկոս",
            "տոկոսադրույք",
            "տարաժամկետ",
        ),
        "branch": (
            "gyumri",
            "yerevan",
            "vanadzor",
            "քաղաք",
            "city",
            "բաց է",
            "աշխատում է",
        ),
    }

    _unsupported_patterns: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bbest bank\b"),
        re.compile(r"\bwhich bank is best\b"),
        re.compile(r"\brecommend a bank\b"),
        re.compile(r"\brecommend\b.*\bbank\b"),
        re.compile(r"\bexchange rate\b"),
        re.compile(r"\bwire transfer\b"),
        re.compile(r"\bmoney transfer\b"),
    )
    _unsupported_substrings_hy: tuple[str, ...] = (
        "փոխարժեք",
        "փոխանցում",
        "քարտի",
        "բանկային քարտ",
        "վճարային քարտ",
    )
    _unsupported_substrings_en: tuple[str, ...] = (
        "exchange rate",
        "transfer money",
        "open a card",
        "payment card",
        "debit card",
        "credit card",
    )
    _injection_terms: tuple[str, ...] = (
        "ignore previous",
        "ignore instructions",
        "system prompt",
        "developer message",
        "bypass policy",
    )

    def _term_hit(self, lower: str, term: str) -> bool:
        t = term.strip().casefold()
        if not t:
            return False
        if t in lower:
            return True
        if self._cfg.fuzzy_match:
            return fuzzy_term_matches(lower, t, ratio=self._cfg.fuzzy_ratio)
        return False

    def _unsupported_hit(self, lower: str) -> bool:
        for pat in self._unsupported_patterns:
            if pat.search(lower):
                return True
        for s in self._unsupported_substrings_hy:
            if s in lower:
                return True
        for s in self._unsupported_substrings_en:
            if s in lower:
                return True
        return False

    def classify(self, query: str) -> TopicClassification:
        lower = query.lower()
        if any(term in lower for term in self._injection_terms):
            return TopicClassification(
                label="unsupported_request_type",
                confidence=1.0,
                reason="prompt_injection_pattern",
            )
        if self._unsupported_hit(lower):
            return TopicClassification(
                label="unsupported_request_type",
                confidence=0.95,
                reason="unsupported_request_intent",
            )

        scores: dict[str, float] = {"credit": 0.0, "deposit": 0.0, "branch": 0.0}
        matched: dict[str, list[str]] = {"credit": [], "deposit": [], "branch": []}
        strong_matched: dict[str, list[str]] = {"credit": [], "deposit": [], "branch": []}

        for topic, terms in self._strong_topic_terms.items():
            for term in terms:
                if self._term_hit(lower, term):
                    scores[topic] += self._cfg.strong_term_weight
                    matched[topic].append(term)
                    strong_matched[topic].append(term)

        for topic, terms in self._weak_topic_terms.items():
            for term in terms:
                if self._term_hit(lower, term):
                    scores[topic] += self._cfg.weak_term_weight
                    matched[topic].append(term)

        any_strong_hit = any(strong_matched[t] for t in scores)

        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]
        if best_score <= 0:
            return TopicClassification(label="out_of_scope", confidence=0.8, reason="no_supported_topic_terms")

        # Weak-only: if one product line clearly dominates weak signals, route there (fewer false "ambiguous").
        if not any_strong_hit:
            ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_t, top_s = ordered[0]
            second_s = ordered[1][1]
            if top_s >= self._cfg.weak_term_weight and (
                second_s == 0 or top_s >= second_s * 1.55
            ):
                return TopicClassification(
                    label=top_t,  # type: ignore[arg-type]
                    confidence=min(0.78, 0.5 + 0.1 * top_s),
                    matched_terms=matched[top_t],
                    reason="weak_signals_single_dominant_topic",
                )
            return TopicClassification(
                label="ambiguous",
                confidence=0.65,
                matched_terms=matched[best_topic],
                reason="weak_topic_signals_only_need_explicit_product",
            )

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        second_score = ordered[1][1]
        only_weak = bool(matched[best_topic]) and not strong_matched[best_topic]
        if second_score > 0 and (best_score - second_score) <= self._cfg.ambiguous_margin and only_weak:
            return TopicClassification(
                label="ambiguous",
                confidence=0.55,
                matched_terms=matched[best_topic] + matched[ordered[1][0]],
                reason="close_topic_scores",
            )
        confidence = min(1.0, 0.55 + 0.15 * best_score)
        return TopicClassification(
            label=best_topic,  # type: ignore[arg-type]
            confidence=confidence,
            matched_terms=matched[best_topic],
            reason="rules_match",
        )
