# Voice AI Banking Support Agent (Offline RAG Pipeline)

This repository contains the **first phase** of an Armenian voice AI banking support agent: an **offline data ingestion + retrieval preparation pipeline**.

It is designed to:
- ingest official bank website pages for **Credits**, **Deposits**, and **Branch Locations** (for 3 initial banks),
- clean and normalize the extracted content,
- parse branch pages into structured records (when possible),
- chunk documents in a section-aware way (RAG-ready),
- compute embeddings using a configurable model,
- store processed artifacts locally and build a local vector index,
- provide a lightweight CLI to inspect retrieval quality.

It intentionally does **not** implement the voice layer (LiveKit) or any online question-answering yet.

## Architecture Overview (High Level)

1. **Manifests / Config** define which URLs are allowed per bank and topic.
2. **Scrapers** fetch HTML for allowed pages (offline, at indexing time only).
3. **Extraction** converts HTML into main text and splits it into sections.
4. **Branch Parsing** attempts to extract structured branch records from branch pages.
5. **Cleaning + Validation** removes noise and skips low-value extractions.
6. **Chunking** creates section-aware text chunks with headings preserved.
7. **Embedding** converts chunks to vectors using a configurable embedding model.
8. **Vector Store** builds and persists a local FAISS index + metadata.
9. **CLI** provides commands for scraping, indexing, and a retrieval demo.

Detailed references:
- `ARCHITECTURE.md`
- `CODE_WALKTHROUGH.md`
- `EXPLAINED_PIPELINE.md`

## Setup

1. Install Python 3.10+.
2. Create and activate a virtual environment.
3. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install -e ".[dev]"`
4. (Optional) If you later want Playwright-based fetching:
   - `pip install -e ".[playwright]"`
5. Copy env template if needed:
   - `.env.example` -> `.env` (optional; used for runtime overrides).

## Project Layout

The code lives under `src/voice_ai_banking_support_agent/` and is intentionally modular.
Manifests are in `manifests/`.

## How to Run Scraping (Dataset Build)

Use the dataset builder pipeline. Examples:

```bash
python -m voice_ai_banking_support_agent.cli scrape ^
  --banks acba ameriabank idbank ^
  --topics credit deposit branch
```

This downloads allowed HTML pages, extracts and cleans content, parses branch records, and writes JSONL artifacts under `data/` (raw HTML, cleaned docs, chunks, branches).

Quality features in current version:
- strict manifest validation
- retries/timeouts/session reuse
- deterministic raw artifact naming
- de-duplicated JSONL appends
- failure ratio guard for safer pipeline runs

## How to Build the Vector Index

```bash
python -m voice_ai_banking_support_agent.cli build-index ^
  --index-name armenian_banks ^
  --banks acba ameriabank idbank ^
  --topics credit deposit branch
```

## How to Run the Retrieval Demo

```bash
python -m voice_ai_banking_support_agent.cli demo-retrieve ^
  --index-name armenian_banks ^
  --query "Որքա՞ն է վարկի տոկոսադրույքը" ^
  --topic credit ^
  --bank acba ^
  --top-k 5
```

The CLI prints retrieved chunks plus their metadata (bank/topic/url/chunk id/section title).

Tip: `--bank` accepts either bank key (`acba`) or display name (`ACBA Bank`).

## Current Limitations

- URL extraction rules and branch-table heuristics are **best-effort**. If a bank changes HTML structure, you may need to adjust parsing heuristics in the relevant scraper/extraction module.
- Retrieval demo uses **embeddings + vector search only** for now. BM25/hybrid is not implemented yet (interfaces are designed to allow later addition).
- Chunking avoids mixing unrelated sections, but you may want to tune chunk sizes based on observed page layouts.
- Ameriabank now uses targeted DNN module API payload extraction for pages where server HTML is a shell. Structured branch records still remain head-office-centric (single-record fallback per URL), so deeper per-branch structured extraction is a remaining improvement area.

## Future Next Step

After this phase is stable, the next step is to connect:
- topic routing (using the retrieved metadata + a future LLM router),
- the online voice layer (LiveKit),
- and answer generation (not built in this phase).

## Runtime Text QA (Next Phase)

Runtime layer is now available for text-only orchestration before voice integration:

- `python cli.py --config validation_manifest_update_hy.yaml runtime-chat --index-name <index_name>`
- `python cli.py --config validation_manifest_update_hy.yaml runtime-eval --index-name <index_name>`
- Optional runtime refinement config:
  - `--runtime-config runtime_config.yaml`
- Optional decision tracing:
  - `--verbose`

What runtime adds:
- rules-first topic control (`credit/deposit/branch` only),
- bank detection + lightweight follow-up resolution,
- topic/bank-filtered retrieval,
- evidence sufficiency checks,
- grounded Armenian responses,
- explicit refusal paths for unsupported/out-of-scope queries.

## LiveKit Voice Integration (Self-hosted)

Voice layer wraps runtime core without duplicating business logic:
- STT -> runtime orchestrator -> TTS
- runtime status/refusal is preserved verbatim in spoken output.
- **Self-hosted/open-source LiveKit only**. LiveKit Cloud must not be used in this project.

### Commands

- Voice smoke (mock STT/TTS, local logic test):
  - `python cli.py --config demo_config.yaml voice-smoke-test --index-name multi_model_index --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --voice-config voice_config.example.yaml`

- Self-hosted LiveKit agent:
  - `python cli.py --config demo_config.yaml voice-agent --index-name multi_model_index --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --voice-config voice_config.example.yaml`

- One-file frontend (local text UI for self-testing):
  - `python frontend_demo.py --config demo_config.yaml --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --index-name multi_model_index --port 8080`
  - open `http://127.0.0.1:8080`

- Minimal React frontend (optional):
  - start API: `python run_runtime_api.py --config demo_config.yaml --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --host 127.0.0.1 --port 8000`
  - start UI: `cd frontend-react && npm install && npm run dev`
  - open Vite URL (usually `http://127.0.0.1:5173`)

- Runtime eval with optional LLM backend:
  - `python cli.py --config demo_config.yaml runtime-eval --index-name multi_model_index --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --verbose`

### LiveKit env/config

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `LIVEKIT_TOKEN` (required for current self-hosted transport loop)

Recommended local self-hosted values:
- `LIVEKIT_URL=ws://127.0.0.1:7880`
- `LIVEKIT_API_KEY=devkey`
- `LIVEKIT_API_SECRET=secret`

### Notes

- Install optional voice dependency when needed:
  - `pip install .[voice]`
- Voice startup validates `LIVEKIT_URL` and refuses LiveKit Cloud domains.
- Voice providers are config-swappable:
  - STT: `mock`, `http_whisper`
  - TTS: `mock`, `http_tts`
- Real provider mode requires self-hosted/local HTTP STT/TTS endpoints.
- LLM mode is configurable via `llm_config.example.yaml`:
  - `provider: mock` for deterministic local tests
  - `provider: openai_compatible_http` for real endpoint mode
- If LLM fails/timeouts/returns unusable output, runtime falls back to extractive safely.
- Current phase provides demo-ready practical integration while preserving runtime safety gates.

## Quick Start Docs

- `DEMO_QUICKSTART.md`
- `FULL_LOCAL_RUN_GUIDE.md`
- `END_TO_END_VALIDATION_REPORT.md`
- `FINAL_SUBMISSION_READINESS.md`

### Self-hosted LiveKit quick setup

1. Install/start LiveKit server locally (self-hosted OSS):
   - `livekit-server --dev`
2. Generate API key/secret (self-hosted):
   - `livekit-server generate-keys`
3. Generate room token for agent run:
   - `lk token create --api-key <KEY> --api-secret <SECRET> --join --room banking-support-room --identity banking-support-agent`
4. Export env:
   - `LIVEKIT_URL=ws://127.0.0.1:7880`
   - `LIVEKIT_API_KEY=<KEY>`
   - `LIVEKIT_API_SECRET=<SECRET>`
   - `LIVEKIT_TOKEN=<TOKEN_FROM_STEP_3>`

---

## CODE WALKTHROUGH
See `CODE_WALKTHROUGH.md` for the maintained file-by-file walkthrough.

## HOW TO EXTEND TO A 4TH BANK

1. Add bank entry to `manifests/banks.yaml` with non-empty URLs for all 3 topics.
2. Create `src/voice_ai_banking_support_agent/scrapers/<new_bank>.py`.
3. Implement:
   - `extraction_rules()`
   - `branch_parsing_hints()`
   - `fetch_structured()` (API/JSON first, DOM fallback).
4. Register scraper in `BANK_SCRAPERS` map inside `build_dataset.py`.
5. Add tests similar to `test_bank_specific_extraction.py` and `test_structured_fetchers.py`.
6. Run: scrape -> build-index -> demo-retrieve, then inspect data artifacts.

## WHAT TO DO NEXT FOR ONLINE QA + LIVEKIT STAGE

Do not build it yet, but prepare this order:
1. Add retrieval-to-answer orchestration with citation-preserving prompt format.
2. Add topic router that leverages metadata filters (`topic`, `bank_key`).
3. Add safety layer (no fabricated rates, fallback to "need human agent").
4. Integrate STT/TTS + session state (LiveKit or equivalent).
5. Add latency budget + observability (query logs, retrieval traces, answer confidence).

### `cli.py` (repo root)
Thin wrapper that lets you run `python cli.py ...` without installing the package.

### `src/voice_ai_banking_support_agent/config.py`
Loads configuration (paths, embedding model, retry/network settings) and validates high-level pipeline settings.

### `src/voice_ai_banking_support_agent/models.py`
Defines typed Pydantic models for documents, chunk metadata, branch records, and topic labels.

### `src/voice_ai_banking_support_agent/bank_manifest.py`
Loads `manifests/banks.yaml` and validates that the manifest schema is correct and that URLs look like HTTP(S) links.

### `src/voice_ai_banking_support_agent/utils/logging.py`
Sets up robust logging with console + optional file handlers.

### `src/voice_ai_banking_support_agent/scrapers/base.py`
Implements the offline fetcher with retries, timeouts, and a basic HTML response container.

### `src/voice_ai_banking_support_agent/scrapers/acba.py`
ACBA-specific branch parsing keyword hints (keeps bank heuristics out of the shared parser).

### `src/voice_ai_banking_support_agent/scrapers/ameriabank.py`
Ameriabank-specific branch parsing keyword hints.

### `src/voice_ai_banking_support_agent/scrapers/idbank.py`
IDBank-specific branch parsing keyword hints.

### `src/voice_ai_banking_support_agent/extraction/cleaning.py`
Main-text extraction + cleaning + normalization.

### `src/voice_ai_banking_support_agent/extraction/section_parser.py`
Turns cleaned HTML content into a list of sections (title + content).

### `src/voice_ai_banking_support_agent/extraction/branch_parser.py`
Attempts to parse structured branch records (name/city/address/hours/phone) from branch pages.

### `src/voice_ai_banking_support_agent/indexing/chunker.py`
Section-aware chunking that preserves headings and keeps related details together.

### `src/voice_ai_banking_support_agent/indexing/embedder.py`
Wraps the embedding model (default: `Metric-AI/armenian-text-embeddings-2-large`) and produces normalized vectors.

### `src/voice_ai_banking_support_agent/indexing/vector_store.py`
Creates/persists a local FAISS index and a JSONL metadata mapping from vector ids to chunk metadata.

### `src/voice_ai_banking_support_agent/pipelines/build_dataset.py`
End-to-end offline ingestion:
fetch allowed pages -> extract -> clean -> parse -> chunk -> write JSONL artifacts.

### `src/voice_ai_banking_support_agent/pipelines/build_index.py`
Reads chunk JSONL artifacts -> embeds -> builds/persists FAISS index.

### `src/voice_ai_banking_support_agent/cli.py`
Provides CLI commands: `scrape`, `build-index`, `demo-retrieve`, `inspect-doc`, `discover-urls`.

### `tests/`
Unit tests for topic validation, cleaning/noise removal, chunking behavior, branch parsing, and manifest loading.

## HOW TO MODIFY THIS PROJECT SAFELY

1. Update `manifests/banks.yaml` first when you want to add/remove pages (avoid scattering URL changes in code).
   - Use `discover-urls` to generate candidate links for manual review before adding them to manifest.
2. Prefer editing extraction heuristics (cleaning/section/branch parsing) instead of altering the pipeline contract.
3. Keep the following interfaces stable:
   - manifest loader output schema,
   - chunk metadata schema,
   - vector store `add()` and `search()` contracts.
4. Add/extend unit tests for any new parsing behavior. This project is designed so parsing regressions are easy to catch locally.

## KNOWN RISKS / NEXT IMPROVEMENTS

- Branch parsing may fail on certain bank pages; improve heuristics by comparing failures and adding targeted parsing rules.
- Some bank pages (notably dynamic service-network map pages) may not expose full branch data in static HTML. In those cases, manual endpoint discovery or browser-network capture may still be required to reach full coverage.
- Embedding quality depends on chunk size and cleanliness; improve chunking after inspecting retrieval outputs.
- Consider adding hybrid retrieval (BM25 + embeddings) later; the architecture is set up to allow it with minimal changes.

## Why Offline Indexed Retrieval Beats Crawling at Query Time

- Bank pages often have rate limits, dynamic content, and frequent layout changes. Crawling during every user query is brittle and slow.
- Offline indexing produces repeatable artifacts, consistent cleaning/chunking, and predictable retrieval latency.
- It also lets you validate extraction quality once (with unit tests + CLI inspection), instead of relying on live crawling under unpredictable conditions.

