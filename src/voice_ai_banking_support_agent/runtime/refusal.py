from __future__ import annotations

from .models import RefusalReason


def refusal_message(reason: RefusalReason) -> str:
    mapping = {
        "out_of_scope": "Կներեք, կարող եմ պատասխանել միայն վարկերի, ավանդների և մասնաճյուղերի մասին հարցերին։",
        "unsupported_request_type": "Կներեք, այս հարցի տեսակը չի սպասարկվում։ Կարող եմ օգնել միայն վարկ, ավանդ կամ մասնաճյուղ թեմաներով։",
        "insufficient_evidence": "Այս պահին բավարար վստահելի տվյալներ չկան հստակ պատասխան տալու համար։ Խնդրում եմ հարցը ճշտել (բանկ, թեմա կամ քաղաք)։",
        "prompt_injection": "Չեմ կարող կատարել այդպիսի հրահանգ։ Կարող եմ օգնել միայն աջակցվող բանկային թեմաներով։",
        "ambiguous": "Հարցը մի փոքր անորոշ է։ Խնդրում եմ նշեք՝ վարկ, ավանդ, թե մասնաճյուղ և ցանկալի բանկը։",
    }
    return mapping[reason]

