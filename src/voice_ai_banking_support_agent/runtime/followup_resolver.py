from __future__ import annotations

from .bank_scope import query_implies_all_banks
from .models import FollowUpResolution, SessionState


class FollowUpResolver:
    """Resolve lightweight follow-up references from session state."""

    def __init__(
        self,
        *,
        followup_markers: list[str] | None = None,
        short_query_max_tokens: int = 5,
        city_terms: list[str] | None = None,
        bank_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self._followup_markers = tuple(
            followup_markers
            or [
                "իսկ",
                "and",
                "what about",
                "and for",
                "also",
                "բայց",
                "իմա",
            ]
        )
        self._short_query_max_tokens = short_query_max_tokens
        self._city_terms = tuple(city_terms or ["երևան", "գյումրի", "gyumri", "yerevan", "վանաձոր", "vanadzor"])
        self._bank_aliases = bank_aliases or {}

    def _topic_terms_in_query(self, lower: str) -> bool:
        hints = (
            "վարկ",
            "ավանդ",
            "մասնաճյուղ",
            "մասնաճյուղեր",
            "սպառողական",
            "loan",
            "loans",
            "credit",
            "deposit",
            "branch",
            "տոկոս",
            "տոկոսադրույք",
            "interest",
            "հասցե",
            "address",
            "որտե",
            "where",
        )
        return any(h in lower for h in hints)

    def _mentions_any_bank(self, lower: str) -> bool:
        for aliases in self._bank_aliases.values():
            for alias in aliases:
                if len(alias) >= 2 and alias.lower() in lower:
                    return True
        return False

    def _query_targets_different_topic_than_state(self, lower: str, last_topic: str) -> bool:
        """True when the user names another product line than the one carried in session (e.g. switches to loans)."""

        credit = ("վարկ", "վարկեր", "սպառողական", "loan", "loans", "mortgage", "credit")
        deposit = ("ավանդ", "deposit", "deposits", "խնայող")
        branch = ("մասնաճյուղ", "մասնաճյուղեր", "branch", "atm", "բանկոմատ", "հասցե", "where", "address", "որտե")
        has_c = any(t in lower for t in credit)
        has_d = any(t in lower for t in deposit)
        has_b = any(t in lower for t in branch)
        if last_topic == "deposit":
            return has_c and not has_d
        if last_topic == "credit":
            return has_d and not has_c
        if last_topic == "branch":
            return (has_c or has_d) and not has_b
        return False

    def resolve(self, query: str, state: SessionState) -> FollowUpResolution:
        q = query.strip()
        lower = q.lower()
        tokens = q.split()
        is_short = len(tokens) <= self._short_query_max_tokens
        is_followup_marker = any(lower.startswith(marker) for marker in self._followup_markers)
        has_reference_hint = any(
            h in lower
            for h in (
                "դեպքում",
                "այդ",
                "that",
                "also",
                "նույն",
                "same",
                "էլ",
                "այլ",
                "ուրիշ",
                "նույնիսկ",
                "այնպես",
                "հետո",
                "նախորդ",
            )
        )
        is_followup = is_followup_marker or (is_short and has_reference_hint)

        # Short bank pivot: "Ameriabank" / "ամերիա" after a prior topic turn.
        if (
            not is_followup
            and state.last_topic
            and len(tokens) <= 8
            and self._mentions_any_bank(lower)
            and not self._topic_terms_in_query(lower)
        ):
            is_followup = True

        if not is_followup:
            return FollowUpResolution(resolved_query=q)

        if is_followup and not (state.last_topic or state.last_bank):
            return FollowUpResolution(resolved_query=q, needs_clarification=True)

        merged_fields: list[str] = []
        prefix_parts: list[str] = []
        explicit_bank_context = " դեպքում" in lower or "for " in lower or "about " in lower
        new_bank_mentioned = self._mentions_any_bank(lower)
        if (
            state.last_bank
            and state.last_bank.lower() not in lower
            and not explicit_bank_context
            and not new_bank_mentioned
            and not query_implies_all_banks(lower)
        ):
            prefix_parts.append(state.last_bank)
            merged_fields.append("last_bank")
        if state.last_topic:
            topic_hint = {"credit": "վարկ", "deposit": "ավանդ", "branch": "մասնաճյուղ"}[state.last_topic]
            if (
                not self._query_targets_different_topic_than_state(lower, state.last_topic)
                and topic_hint not in lower
                and state.last_topic not in lower
            ):
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
