# Production Hardening Updates

This version adds practical production hardening without overengineering:

- Strict manifest validation:
  - schema version check,
  - unique `bank_key`,
  - non-empty topic URL lists,
  - URL de-duplication.
- Fetching safety:
  - reusable HTTP session,
  - retry/backoff still enabled,
  - explicit response checks for HTTP status, content type, and body length,
  - targeted Ameriabank DNN module payload fetch (`API/WebsitesCreative/MyContentManager/API/Init`) for pages where server HTML is mostly shell placeholders.
- Dataset idempotency:
  - JSONL appends now de-duplicate records by stable keys (`chunk_id`, page key, branch key).
- Metadata consistency:
  - chunk metadata now stores both `bank_key` and `bank_name`.
  - retrieval filter supports either.
- Failure transparency:
  - per-page errors are logged,
  - run-level failure ratio is reported,
  - pipeline fails if too many pages break in one run.
- Index safety:
  - index and metadata write with temp file replacement,
  - runtime check ensures FAISS vector count equals metadata row count.

# Explained Pipeline (Offline Ingestion -> Retrieval-Ready Index)

This document describes the pipeline in very plain terms, module-by-module, so you can understand and improve it later.

## Big Picture

At a high level, the system does this once (offline) for each bank/topic:
1. Downloads allowed web pages (HTML) from the bank website.
2. Extracts the *main content* text from the HTML, removing menus/footer noise.
3. Cleans that text into a normalized form.
4. If the topic is `branch`, tries to extract structured branch records (address, hours, phone, etc).
5. Splits the page content into smaller RAG chunks in a “section-aware” way.
6. Turns each chunk into an embedding vector.
7. Stores:
   - cleaned chunk documents (JSONL),
   - branch records (JSONL),
   - a local FAISS vector index + metadata mapping.
8. Provides a CLI to query the index and inspect results.

User-time retrieval is then fast and predictable: query text is embedded and compared to stored chunk vectors.

## Module-by-Module

### `src/voice_ai_banking_support_agent/config.py`
Purpose:
- Centralizes all configuration: where to read/write data, which embedding model to use, network timeouts/retries, etc.

Why it exists:
- Keeps scraping/indexing code from hardcoding file paths and model names all over the project.

Inputs:
- Optional YAML config file (and environment variables later if you extend it).

Outputs:
- A strongly-typed `AppConfig` object used by pipelines and CLI.

### `src/voice_ai_banking_support_agent/models.py`
Purpose:
- Holds typed data structures using Pydantic.

Key types:
- Topic labels: must be exactly one of `credit`, `deposit`, `branch`.
- `DocumentMetadata`: bank/topic/url/title/language/chunk id, plus the raw + cleaned text fields.
- `BranchRecord`: structured output for branch parsing.

Why it exists:
- Validation early prevents “garbage in garbage out”.
- It also makes it easy to extend metadata later.

### `src/voice_ai_banking_support_agent/utils/logging.py`
Purpose:
- Consistent logging configuration across the whole project.

Why it exists:
- Scraping/indexing pipelines are long-running; you need reliable logs and warnings.

### `src/voice_ai_banking_support_agent/bank_manifest.py`
Purpose:
- Loads and validates the bank source manifest from `manifests/banks.yaml`.

Why it exists:
- You explicitly define allowed pages per topic per bank.
- This prevents accidentally scraping entire sites.

Inputs:
- YAML manifest file.

Outputs:
- Typed manifest objects used by the dataset builder.

### `src/voice_ai_banking_support_agent/scrapers/base.py`
Purpose:
- Offline HTML fetching with retries/timeouts, and basic HTTP headers.

Why it exists:
- Keeps network concerns (retries, timeouts, user-agent) out of extraction/chunking logic.

Implementation detail:
- Default fetcher uses `requests`.
- If later you need JS rendering, an optional Playwright fetcher can be added behind a config flag.

### `src/voice_ai_banking_support_agent/extraction/cleaning.py`
Purpose:
- Converts raw HTML into “main text” and normalizes it.

What it does:
- Removes script/style/nav/footer/header/aside-like content.
- Collapses whitespace.
- Applies basic normalization for punctuation and line breaks.
- Validates that the text is not empty/noisy.

Outputs:
- `cleaned_text` and warnings when content is too small/low-value.

### `src/voice_ai_banking_support_agent/extraction/section_parser.py`
Purpose:
- Splits cleaned content into sections based on HTML headings (`h1`, `h2`, `h3` etc).

Why it exists:
- Naive fixed-size chunking loses the structure needed for RAG.
- Section-aware chunking preserves topic context and headings.

Outputs:
- A list of `Section` objects: each section has a title + associated content.

### `src/voice_ai_banking_support_agent/extraction/branch_parser.py`
Purpose:
- Tries to extract structured branch records from branch pages.

What it does (best-effort heuristics):
- Looks for tables/lists where columns/cells match:
  - branch name
  - city
  - address (must exist ideally)
  - working hours
  - phone
- Also uses regex to detect phone numbers and address-like patterns in text.

Validation:
- If address is missing, it logs a warning.
- If it finds zero useful records, it returns an empty list and logs a warning.

Outputs:
- A list of `BranchRecord` objects.

### `src/voice_ai_banking_support_agent/indexing/chunker.py`
Purpose:
- Turns sections into smaller chunks suitable for embedding + retrieval.

Chunking strategy:
- Keeps each chunk inside a single section.
- Preserves the section heading with each chunk.
- Uses a “target chunk size” in words/chars so chunks are not too large/small.
- Tries to avoid splitting short numeric details across boundaries.

Outputs:
- Chunk objects with unique `chunk_id`, `section_title`, and `cleaned_text`.

### `src/voice_ai_banking_support_agent/indexing/embedder.py`
Purpose:
- Produces embeddings for chunks.

Key design choices:
- Embedding model name is configurable.
- Default model: `Metric-AI/armenian-text-embeddings-2-large`.
- Embeddings are normalized for cosine similarity.

Outputs:
- A matrix of vectors and any embedding configuration metadata you want to store.

### `src/voice_ai_banking_support_agent/indexing/vector_store.py`
Purpose:
- Persists a local FAISS index and stores metadata mapping.

Why it exists:
- Your retrieval demo needs to load the index quickly from disk.
- Metadata (bank/topic/url/chunk id/raw_text/cleaned_text) is required to interpret results.

Outputs:
- `faiss.index` file and JSONL mapping files under `data/index/<index_name>/`.

### `src/voice_ai_banking_support_agent/pipelines/build_dataset.py`
Purpose:
- End-to-end offline dataset builder.

Step-by-step:
1. Load the manifest.
2. For each bank/topic:
   - For each allowed URL:
     - Fetch HTML (offline).
     - Save raw HTML to disk.
     - Extract and clean main text.
     - If branch topic: parse structured branches.
     - Parse headings into sections.
     - Chunk sections into RAG documents.
     - Write:
       - raw HTML artifacts
       - cleaned chunk JSONL
       - branch JSONL (for branch topic)

### `src/voice_ai_banking_support_agent/pipelines/build_index.py`
Purpose:
- Builds the vector index from the chunk JSONL artifacts.

Step-by-step:
1. Read chunks JSONL.
2. Embed chunk texts.
3. Add embeddings to FAISS index.
4. Persist FAISS + metadata.

### `src/voice_ai_banking_support_agent/cli.py`
Purpose:
- A convenient interface to run scraping/indexing and test retrieval quality.

Commands:
- `scrape`: runs dataset build for chosen banks/topics.
- `build-index`: builds FAISS index.
- `demo-retrieve`: embeds query text and prints top results with metadata.
- `inspect-doc`: shows a stored chunk by chunk id.

## What Data Flows Between Modules

1. `bank_manifest.py` -> list of allowed URLs (per bank/topic).
2. `scrapers/base.py` -> raw HTML (string) + HTTP info.
3. `extraction/cleaning.py` -> cleaned main text.
4. `extraction/section_parser.py` -> structured sections (title/content).
5. `extraction/branch_parser.py` -> branch records (structured fields).
6. `indexing/chunker.py` -> chunk documents (chunk_id, section title, text).
7. `indexing/embedder.py` -> embeddings vectors.
8. `indexing/vector_store.py` -> FAISS index + metadata mapping.
9. `cli.py` retrieval -> vector search -> printed metadata/chunks.

## Why This Design Is Scalable

- Adding banks is mostly adding a new scraper + manifest entries (URLs).
- Metadata is consistent across documents; retrieval can filter by bank/topic later.
- Chunking preserves section context, which tends to produce higher-quality retrieval.
- The retriever interface can later expand to hybrid retrieval (BM25 + embeddings) without rewriting ingestion.

## CODE WALKTHROUGH

See `README.md` for a short code walkthrough. This file focuses on the pipeline behavior.

## HOW TO MODIFY THIS PROJECT SAFELY

1. Edit `manifests/banks.yaml` when you change allowed URLs (not scattered in code).
2. Tune extraction by updating `extraction/cleaning.py`, `extraction/section_parser.py`, or `extraction/branch_parser.py`.
3. Tune chunking by adjusting `indexing/chunker.py` target chunk size.
4. Tune embedding by changing config `embedding_model`.
5. Run unit tests before and after parsing heuristic changes.

## KNOWN RISKS / NEXT IMPROVEMENTS

- Branch parsing heuristics are best-effort. Some pages may have unconventional tables.
- Chunk size may need tuning for each bank's page structure.
- Embeddings may require multilingual baseline comparisons later.

## Why Offline Indexed Retrieval Beats Crawling at Query Time

Offline indexing avoids:
- rate limits and network instability during user queries,
- unpredictable HTML/JS rendering changes during live answering,
- slow latency from fetching and re-processing pages every request.

Instead, you do the expensive work once, and you get stable retrieval quality and fast responses.

