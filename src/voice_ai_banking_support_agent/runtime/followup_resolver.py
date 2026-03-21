from __future__ import annotations

from .models import FollowUpResolution, SessionState


class FollowUpResolver:
    """Resolve lightweight follow-up references from session state."""

    def __init__(
        self,
        *,
        followup_markers: list[str] | None = None,
        short_query_max_tokens: int = 5,
        city_terms: list[str] | None = None,
    ) -> None:
        self._followup_markers = tuple(followup_markers or ["իսկ", "and", "what about", "and for", "also"])
        self._short_query_max_tokens = short_query_max_tokens
        self._city_terms = tuple(city_terms or ["երևան", "գյումրի", "gyumri", "yerevan"])

    def resolve(self, query: str, state: SessionState) -> FollowUpResolution:
        q = query.strip()
        lower = q.lower()
        tokens = q.split()
        is_short = len(tokens) <= self._short_query_max_tokens
        is_followup_marker = any(lower.startswith(marker) for marker in self._followup_markers)
        has_reference_hint = any(h in lower for h in ("դեպքում", "այդ", "that", "also"))
        is_followup = is_followup_marker or (is_short and has_reference_hint)
        if not is_followup:
            return FollowUpResolution(resolved_query=q)

        merged_fields: list[str] = []
        prefix_parts: list[str] = []
        explicit_bank_context = " դեպքում" in lower or "for " in lower or "about " in lower
        if state.last_bank and state.last_bank.lower() not in lower and not explicit_bank_context:
            prefix_parts.append(state.last_bank)
            merged_fields.append("last_bank")
        if state.last_topic:
            topic_hint = {"credit": "վարկ", "deposit": "ավանդ", "branch": "մասնաճյուղ"}[state.last_topic]
            if topic_hint not in lower and state.last_topic not in lower:
                prefix_parts.append(topic_hint)
                merged_fields.append("last_topic")
        query_has_city = any(city in lower for city in self._city_terms)
        if state.last_city and state.last_city.lower() not in lower and not query_has_city:
            prefix_parts.append(state.last_city)
            merged_fields.append("last_city")
        if state.last_product and state.last_product.lower() not in lower and any(
            p in lower for p in ("դեպքում", "about", "իսկ")
        ):
            prefix_parts.append(state.last_product)
            merged_fields.append("last_product")

        if not prefix_parts:
            return FollowUpResolution(resolved_query=q)
        return FollowUpResolution(
            resolved_query=f"{' '.join(prefix_parts)} {q}".strip(),
            used_followup_context=True,
            merged_fields=merged_fields,
        )

