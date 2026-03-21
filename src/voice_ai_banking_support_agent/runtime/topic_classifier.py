from __future__ import annotations

from dataclasses import dataclass

from .models import TopicClassification


@dataclass(frozen=True)
class TopicClassifierConfig:
    ambiguous_margin: float = 0.15
    strong_term_weight: float = 1.5
    weak_term_weight: float = 0.75


class TopicClassifier:
    """
    Rules-first topic and request-type classifier.

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
            "սպառողական",
            "loan",
            "loans",
            "mortgage",
            "credit",
        ),
        "deposit": (
            "ավանդ",
            "խնայող",
            "deposit",
            "deposits",
        ),
        "branch": (
            "մասնաճյուղ",
            "բանկոմատ",
            "հասցե",
            "որտե՞ղ",
            "where",
            "address",
            "branch",
            "atm",
        ),
    }
    _weak_topic_terms: dict[str, tuple[str, ...]] = {
        "credit": ("interest rate", "տոկոս", "monthly"),
        "deposit": ("interest", "rate", "dollar", "dram", "currency", "cumulative deposit", "տոկոս"),
        "branch": ("gyumri", "yerevan", "vanadzor", "քաղաք", "city"),
    }

    _unsupported_terms: tuple[str, ...] = (
        "best bank",
        "which bank is best",
        "recommend",
        "խորհուրդ",
        "քարտ",
        "card",
        "cards",
        "exchange rate",
        "փոխարժեք",
        "transfer",
        "փոխանցում",
    )
    _injection_terms: tuple[str, ...] = (
        "ignore previous",
        "ignore instructions",
        "system prompt",
        "developer message",
        "bypass policy",
    )

    def classify(self, query: str) -> TopicClassification:
        lower = query.lower()
        if any(term in lower for term in self._injection_terms):
            return TopicClassification(
                label="unsupported_request_type",
                confidence=1.0,
                reason="prompt_injection_pattern",
            )
        if any(term in lower for term in self._unsupported_terms):
            return TopicClassification(
                label="unsupported_request_type",
                confidence=0.95,
                reason="unsupported_request_intent",
            )

        scores: dict[str, float] = {"credit": 0.0, "deposit": 0.0, "branch": 0.0}
        matched: dict[str, list[str]] = {"credit": [], "deposit": [], "branch": []}
        for topic, terms in self._strong_topic_terms.items():
            for term in terms:
                if term in lower:
                    scores[topic] += self._cfg.strong_term_weight
                    matched[topic].append(term)
        for topic, terms in self._weak_topic_terms.items():
            for term in terms:
                if term in lower:
                    scores[topic] += self._cfg.weak_term_weight
                    matched[topic].append(term)

        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]
        if best_score <= 0:
            return TopicClassification(label="out_of_scope", confidence=0.8, reason="no_supported_topic_terms")

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        second_score = ordered[1][1]
        only_weak = bool(matched[best_topic]) and all(
            t in self._weak_topic_terms[best_topic] for t in matched[best_topic]
        )
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

