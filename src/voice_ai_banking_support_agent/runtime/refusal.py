from __future__ import annotations

from .models import RefusalReason
from .rag_prompts import REFUSAL_RULES


def bank_clarification_message(allowed_bank_keys_csv: str) -> str:
    """Ask user to name a bank or ask for all banks (strict orchestration)."""

    return REFUSAL_RULES["clarify_bank"].format(banks=allowed_bank_keys_csv)


def refusal_message(reason: RefusalReason) -> str:
    """User-facing Armenian copy; bank-agnostic where possible (see REFUSAL_RULES)."""

    extra: dict[RefusalReason, str] = {
        "unsupported_request_type": (
            "Այս հարցը թեմայից դուրս է։ Օգնում եմ միայն վարկ, ավանդ և մասնաճյուղ/հասցե հարցերով։"
        ),
        "prompt_injection": (
            "Չեմ կարող կատարել այդ հրահանգը։ Օգնում եմ միայն վարկ, ավանդ և մասնաճյուղ թեմաներով։"
        ),
        "ambiguous": REFUSAL_RULES["clarify_topic"],
    }
    if reason in extra:
        return extra[reason]
    if reason == "out_of_scope":
        return REFUSAL_RULES["out_of_scope"]
    if reason == "insufficient_evidence":
        return REFUSAL_RULES["insufficient_evidence"]
    if reason == "comparison_insufficient":
        return REFUSAL_RULES["comparison_insufficient"]
    raise ValueError(f"Unhandled refusal reason: {reason!r}")
