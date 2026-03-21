# Runtime Architecture

## Goal

This runtime is the text QA orchestration layer that sits on top of your existing offline corpus/index.
It answers only supported Armenian banking topics:

- `credit`
- `deposit`
- `branch`

It explicitly refuses unsupported/out-of-scope queries.

## Step-by-step runtime pipeline

1. **Input**  
   Receive `session_id`, `query`, `index_name`.

2. **Normalization**  
   Normalize whitespace and query shape (`runtime/query_normalizer.py`).

3. **Follow-up resolution**  
   If the query looks like a follow-up (`իսկ...`, `and...`, short turn), merge lightweight context from session state (`runtime/followup_resolver.py`).

4. **Topic/scope classification**  
   Rules-first classifier (`runtime/topic_classifier.py`) assigns one of:
   - `credit`, `deposit`, `branch`
   - `out_of_scope`
   - `ambiguous`
   - `unsupported_request_type`

5. **Bank detection**  
   Bank aliases in Armenian/English are matched (`runtime/bank_detector.py`).

6. **Code-level refusal gate**  
   If class is out-of-scope/unsupported/ambiguous, runtime refuses immediately (`runtime/refusal.py`) without retrieval/generation.

7. **Retrieval orchestration**  
   `runtime/retriever.py`:
   - embeds query with existing embedding stack
   - searches existing FAISS index
   - applies topic/bank filtering through existing vector store logic

8. **Evidence sufficiency check**  
   `runtime/evidence_checker.py` validates evidence quality before answer generation.
   Example checks:
   - no chunks
   - top score below threshold
   - branch address question without address-like evidence

9. **Grounded answer generation**  
   `runtime/answer_generator.py` generates Armenian answer text from retrieved chunks only (deterministic extractive style for safety).

10. **Structured output**  
    `RuntimeResponse` includes:
    - answer text
    - status (`answered/refused/clarify`)
    - refusal reason (if any)
    - detected topic/bank
    - used sources
    - retrieved chunk summary
    - state updates

11. **Session state update**  
    Update `last_topic`, `last_bank`, recent turns (`runtime/session_state.py`).

## Why this design

- **Rules-first control** avoids fragile prompt-only routing.
- **Code-level refusals** enforce safety and scope constraints deterministically.
- **Retrieval before generation** ensures grounding.
- **Evidence gate** reduces hallucinations.
- **Dependency injection in orchestrator** keeps components swappable and testable.

## How this connects to LiveKit later

`RuntimeOrchestrator` is transport-agnostic:
- today: CLI/API text
- later: LiveKit voice transport can feed transcribed text into same `handle()` method and read structured response.

No core logic rewrite is needed; only input/output transport changes.
