from voice_ai_banking_support_agent.runtime.intent_llm import (
    build_intent_classification_prompts,
    parse_intent_classification_json,
)


def test_parse_intent_json_strips_fence() -> None:
    raw = '```json\n{"intent": "deposit", "banks_mentioned": [], "wants_all_banks": false}\n```'
    out = parse_intent_classification_json(raw)
    assert out is not None
    assert out.get("intent") == "deposit"


def test_build_intent_prompts_includes_catalog() -> None:
    sys_p, user_p = build_intent_classification_prompts("Ինչ ավանդներ կան", ["acba", "idbank"])
    assert "acba" in user_p and "idbank" in user_p
    assert "Ինչ ավանդներ կան" in user_p
    assert sys_p
