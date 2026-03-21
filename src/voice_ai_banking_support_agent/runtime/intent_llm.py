"""
Optional LLM-based intent classification (JSON only).

The default pipeline uses rules (TopicClassifier + BankDetector). Call these helpers when you
want a second opinion or to shadow-test classifier agreement without changing production gates.
"""

from __future__ import annotations

import json
import re

from .rag_prompts import (
    INTENT_CLASSIFIER_SYSTEM,
    INTENT_CLASSIFIER_USER_TEMPLATE,
    format_bank_catalog_for_intent,
)


def build_intent_classification_prompts(user_message: str, bank_keys: list[str]) -> tuple[str, str]:
    catalog = format_bank_catalog_for_intent(bank_keys)
    user = INTENT_CLASSIFIER_USER_TEMPLATE.format(bank_catalog=catalog, user_message=user_message.strip())
    return INTENT_CLASSIFIER_SYSTEM.strip(), user


def parse_intent_classification_json(raw: str) -> dict | None:
    """Strip optional ```json fences and parse; return None on failure."""

    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None
