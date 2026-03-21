# Prompt architecture (RAG + voice)

## Source of truth

| Piece | Module |
|--------|--------|
| English system instruction (Gemini) | `runtime/rag_prompts.py` → `RAG_ANSWER_SYSTEM_MESSAGE_EN` (re-exported as `runtime/llm.py` `RAG_SYSTEM_MESSAGE`) |
| Armenian user preamble + footnote slot | `VOICE_ANSWER_PREAMBLE` + `prompts.py` `STANDARD_AI_FOOTNOTE_LINE` |
| Comparison / single / multi-bank supplements | `COMPARISON_PROMPT`, `MULTI_BANK_NON_COMPARE_PROMPT`, `SINGLE_BANK_PROMPT` |
| Optional LLM intent (JSON only) | `INTENT_CLASSIFIER_*`, `intent_llm.py` helpers |
| Refusal / clarification copy | `REFUSAL_RULES`, `refusal.py` |
| Backend gates (not in prompts) | `TopicClassifier`, `BankDetector`, `EvidenceChecker`, `OrchestratorSettings` in `runtime_config.yaml` |

## Orchestration flags (`orchestration:`)

- `require_explicit_bank` — clarify before retrieval if no bank and not “all banks” / comparison.
- `restrict_evidence_to_single_bank_without_comparison` — collapse retrieval to dominant bank when unscoped.
- `clarify_when_unscoped_multi_bank_evidence` — if retrieval spans ≥2 banks but query is unscoped, clarify (no LLM).
- `refuse_comparison_without_multi_bank_evidence` — comparison queries need ≥2 banks in evidence or `comparison_insufficient`.

## Tests

- `tests/test_intent_llm.py` — JSON parse / prompt build.
- `tests/test_orchestration_strict_gates.py` — strict flags above.
