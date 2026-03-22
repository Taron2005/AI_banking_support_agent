"""
Heuristics to sharpen retrieval queries and LLM instructions from user wording.

Keeps logic out of the orchestrator body — Armenian + English cues for deposits/credits.
"""

from __future__ import annotations


def retrieval_query_with_topic_boost(effective_query: str, *, runtime_topic: str) -> str:
    """Append disambiguating tokens for embedding when the user names a deposit subtype."""

    if runtime_topic != "deposit":
        return effective_query
    low = effective_query.lower()
    extra: list[str] = []
    if any(t in low for t in ("ժամկետային", "ֆիքսված ժամկետ", "term deposit", "fixed deposit")):
        extra.append("ժամկետային ֆիքսված ժամկետի ավանդ")
    if any(
        t in low
        for t in (
            "ցպահանջ",
            "անժամկետ",
            "պահանջվող",
            "on demand",
            "demand deposit",
            "checking",
        )
    ):
        extra.append("ցպահանջ անժամկետ պահանջվող ավանդ")
    if any(t in low for t in ("մանկական", "երեխաների", "children", "child")):
        extra.append("մանկական ավանդ երեխաների")
    if not extra:
        return effective_query
    return f"{effective_query.strip()} {' '.join(extra)}".strip()


def extra_llm_context(effective_query: str, *, runtime_topic: str) -> str | None:
    """Short Armenian notes injected into conversation context for Gemini (not evidence)."""

    low = effective_query.lower()
    parts: list[str] = []

    if runtime_topic == "deposit":
        is_compare = any(
            t in low for t in ("համեմատ", "compare", "versus", " vs ", "դեմ համեմատ", "ով է ավելի")
        )
        if is_compare:
            has_term = any(t in low for t in ("ժամկետային", "ֆիքսված ժամկետ"))
            has_demand = any(t in low for t in ("ցպահանջ", "անժամկետ", "պահանջվող"))
            if has_term:
                parts.append(
                    "Համեմատության ֆոկուս․ հարցը վերաբերում է «ժամկետային (ֆիքսված ժամկետի)» ավանդներին։ "
                    "Չխառնել ցպահանջ/անժամկետ ավանդների տոկոսների հետ։ "
                    "Եթե մի բանկի համար ժամկետային տվյալ չկա ապացույցում, նշիր դա առանձին, մի փոխարինիր պահանջվող ավանդով։"
                )
            elif has_demand:
                parts.append(
                    "Համեմատության ֆոկուս․ հարցը վերաբերում է «ցպահանջ/անժամկետ» ավանդներին։ "
                    "Չխառնել ժամկետային ավանդի առավելագույն տոկոսների հետ։"
                )
            else:
                parts.append(
                    "Համեմատության ֆոկուս․ ավանդների տոկոսների հարցում առանձին ներկայացրու ժամկետային, "
                    "ցպահանջ/անժամկետ և մանկական տողերը՝ մի համարիր տարբեր տիպերի տոկոսները մեկ «լավագույն» թվի մեջ։"
                )

    if runtime_topic == "credit":
        if any(t in low for t in ("ավտովարկ", "ավտո վարկ")) or (
            "auto" in low and "loan" in low
        ):
            parts.append(
                "Հարցի ֆոկուս․ ավտովարկ (մեքենայի վարկ)։ Եթե ապացույցում ավտովարկի մասին տվյալ չկա, "
                "միայն ասա՝ ապացույցում չի գտնվել, և մի թվարկիր այլ վարկատեսակների (սպառողական, հիփոթեք, գրավադրամ) "
                "նվազագույն կամ առավելագույն գումարներ կամ պայմաններ։"
            )
        elif "հիփոթեք" in low or "mortgage" in low:
            parts.append(
                "Հարցի ֆոկուս․ հիփոթեքային/բնակարանային վարկ։ Եթե ապացույցում համապատասխան տող չկա, "
                "չփոխարինիր սպառողական կամ ավտովարկի թվերով։"
            )

    if not parts:
        return None
    return "\n".join(parts)
