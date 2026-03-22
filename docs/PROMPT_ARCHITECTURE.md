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

Committed **`runtime_config.yaml`** defaults (production-oriented):

- `restrict_evidence_to_single_bank_without_comparison: true` — for queries that are **not** “all banks” and **not** an explicit comparison, collapse retrieved chunks to the **dominant bank** (highest-scoring bucket) so diversify/rerank cannot mix competitors into one answer.
- `clarify_when_unscoped_multi_bank_evidence: true` — if retrieval still spans **≥2 banks** while the query is unscoped, return **clarify** (no LLM) instead of a blended answer.
- `refuse_comparison_without_multi_bank_evidence: true` — if the user uses explicit comparison wording but post-filter evidence contains **<2 banks**, return **`comparison_insufficient`**.
- `require_explicit_bank` — default **false** (keeps Armenian bank names that are implied but not strictly “explicit”); set **true** for stricter eval.

## Evidence scope (backend, not prompts)

After vector search, **`filter_chunks_to_bank_keys`** in `runtime/evidence_select.py` **post-filters** hits when `bank_keys` is set (one or more named banks). API field **`used_sources`** is built only from **`prepare_evidence_for_answer`** output (the same pool as the LLM), not from the pre-dedupe full retrieval list — so competitor URLs do not appear when only one bank is in scope.

## Query sharpening (retrieval + LLM context)

- **`runtime/query_answer_hints.py`**: `retrieval_query_with_topic_boost` appends Armenian tokens for **deposit subtype** (e.g. ժամկետային vs ցպահանջ) to the embedding query so FAISS/BM25 favor the right pages. `extra_llm_context` adds short Armenian **focus lines** (comparison deposit splits, narrow car-loan / mortgage questions) into the conversation block passed to Gemini — not treated as factual evidence.
- **`bank_scope.ALL_BANKS_QUERY_PHRASES`**: includes **«բոլոր ավանդ…»** so requests like “links for all deposit pages” count as **all-banks** scope and skip the unscoped multi-bank clarify gate.
- **Branch evidence**: `evidence_checker` accepts chunks that look like **branch listings** (`մասնաճյուղ` + `ք.` / `շենք` / `պող.` / `+374`, etc.) when the user asks for addresses, so IDBank-style pages without the literal word «հասցե» still pass the gate. Extra patterns are also listed under `evidence.branch_address_patterns` in `runtime_config.yaml`.

## Tests

- `tests/test_intent_llm.py` — JSON parse / prompt build.
- `tests/test_orchestration_strict_gates.py` — strict flags above.
