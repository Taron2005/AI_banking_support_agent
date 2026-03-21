# OWNER GUIDE

This guide is written for you as the project owner. It explains the pipeline in plain language but with enough technical detail to let you safely extend and debug it.

---

## 1) What this project does (in one sentence)

It builds an **offline, inspectable knowledge base** from Armenian bank websites (credits, deposits, branches), converts that knowledge into chunks + embeddings, and lets you test retrieval quality locally.

---

## 2) End-to-end pipeline in plain language

### Step 0: Manifest controls scope
- File: `manifests/banks.yaml`
- You list exactly which URLs are allowed for each bank/topic.
- Why: predictable ingestion, no uncontrolled crawling.

### Step 1: Load config and manifest
- Config from `config.py` (`AppConfig`, `NetworkConfig`, `ChunkingConfig`).
- Manifest is validated by `bank_manifest.py`.
- Why: fail early on bad setup instead of failing deep in the pipeline.

### Step 2: Fetch HTML and save raw artifacts
- `RequestsHTMLFetcher.fetch()` downloads each URL.
- Raw HTML is saved under `data/raw_html/<bank>/<topic>/...html`.
- Why: reproducibility + easy debugging when extraction fails.

### Step 3: Clean and normalize page text
- `clean_html_to_text()` removes navigation/footer/cookie/chat noise.
- Applies bank-specific extraction rules from scraper modules.
- Why: improves chunk quality and embedding signal.

### Step 4: Language hint + page-level artifact
- `detect_language_from_text()` estimates dominant language.
- Cleaned page record is saved to `data/cleaned_docs/...jsonl`.
- Why: auditing (you can inspect what model actually sees).

### Step 5: Branch structured extraction (branch topic only)
- First try bank-specific `fetch_structured()` (API/JSON-first).
- If empty, fallback to `parse_branch_records()` table/text parser.
- Save records to `data/branches/<bank>_branches.jsonl`.
- Why: branch data is often better as structured fields than plain text.

### Step 6: Section parsing
- `parse_sections_from_html()` splits content into semantic sections by headings.
- Fallback block parsing is used for card/list-heavy pages.
- Why: prevent mixing unrelated page parts in one chunk.

### Step 7: Chunking
- `chunk_sections()` creates section-aware chunks with heading context.
- Save to `data/chunks/...jsonl`.
- Why: chunk quality drives retrieval quality.

### Step 8: Embedding + index
- `build_index()` reads chunk JSONL.
- `EmbeddingModel` converts chunk text to vectors.
- `FaissVectorStore.build_and_save()` writes FAISS index + metadata.
- Why: fast local semantic retrieval.

### Step 9: Retrieval demo
- `demo-retrieve` embeds query and searches FAISS.
- Optional filters: topic and bank.
- Prints score + metadata + chunk preview.
- Why: quick quality check before integrating an LLM.

---

## 3) Core classes/functions you should know first

## Configuration and schema
- `config.py`
  - `AppConfig`: all important paths + model name.
  - `NetworkConfig`: timeout/retries/user-agent.
  - `ChunkingConfig`: target/min/max words.
  - `load_config()`: YAML/env overrides.

- `models.py`
  - `DocumentMetadata` (chunk record contract).
  - `BranchRecord` (structured branch contract).
  - `BanksManifest` and related manifest models.

- `bank_manifest.py`
  - `load_banks_manifest()`: strict manifest validation.
  - `manifest_summary()`: run-time visibility.

## Scraping and extraction
- `scrapers/base.py`
  - `RequestsHTMLFetcher.fetch()`: robust HTML fetch.
  - `RequestsHTMLFetcher.fetch_json()`: API probing helper.
  - `parse_json_ld_objects()`, `parse_inline_json_objects()`.

- `scrapers/acba.py`, `ameriabank.py`, `idbank.py`
  - `extraction_rules()`: bank-specific selector strategy.
  - `fetch_structured()`: API/JSON-first structured extraction hook.

- `extraction/cleaning.py`
  - `clean_html_to_text()`: main cleaner.
  - `detect_language_from_text()`.

- `extraction/section_parser.py`
  - `parse_sections_from_html()`.

- `extraction/branch_parser.py`
  - `parse_branch_records()`: table-first, text fallback.

## Chunking and indexing
- `indexing/chunker.py`
  - `chunk_sections()`: section-aware chunk builder.

- `indexing/embedder.py`
  - `EmbeddingModel`: model loading and batch embedding.

- `indexing/vector_store.py`
  - `FaissVectorStore.build_and_save()`: persist index.
  - `FaissVectorStore.search()`: retrieval + post-filtering.

## Pipelines
- `pipelines/build_dataset.py`
  - `build_dataset()`: orchestration of ingestion stages.
  - `_DedupJsonlAppender.append()`: idempotent JSONL output behavior.

- `pipelines/build_index.py`
  - `build_index()`: chunk loading, dedupe, embedding, index write.

- `pipelines/discover_urls.py`
  - `discover_urls()`: controlled same-domain candidate URL discovery.

---

## 4) Stage-by-stage input/output contracts

### Manifest stage
- Input: `manifests/banks.yaml`
- Output: validated `BanksManifest` object
- Breaks if: duplicate bank keys, empty URL lists, bad schema version, invalid URLs

### Fetch stage
- Input: URL + `NetworkConfig`
- Output: `HTMLFetchResult` + saved HTML file
- Breaks if: non-HTML response / 4xx/5xx / short body

### Clean stage
- Input: raw HTML + bank extraction rules
- Output: cleaned page text (`cleaned_text`) + warning flag
- Breaks if: selectors too aggressive and remove real content

### Branch parse stage
- Input: HTML + cleaned text + branch hints/structured fetcher
- Output: `BranchRecord` list
- Breaks if: dynamic site hides branch data and no usable API/JSON is exposed

### Section+chunk stage
- Input: parsed sections + chunking config
- Output: `DocumentMetadata` chunk rows
- Breaks if: section parser finds nothing or content too noisy/short

### Index stage
- Input: chunk JSONL
- Output: FAISS index + metadata JSONL + index_info JSON
- Breaks if: malformed chunk rows dominate or embedding model fails

### Retrieval stage
- Input: query + built index + optional filters
- Output: ranked retrieval hits
- Breaks if: bad chunk quality, wrong filters, low corpus coverage

---

## 5) Why major design choices were made

- **Manifest-driven ingestion**: you control precision and avoid random crawl noise.
- **Raw artifact persistence**: every extraction bug can be traced to raw HTML.
- **Bank-specific rules + shared pipeline**: reuse where possible, customize only where needed.
- **Section-aware chunking**: retrieval relevance is better than fixed blind splits.
- **Structured branches + text fallback**: best of both worlds for branch QA.
- **FAISS local index**: fast offline evaluation and deterministic demo behavior.
- **De-duplicated JSONL writes**: safer repeated runs while keeping artifact transparency.

---

## 6) Where you can safely modify code

### Safest places
- `manifests/banks.yaml` (URL coverage tuning).
- `scrapers/<bank>.py` selector rules and `fetch_structured()`.
- `ChunkingConfig` values in `config.py` or YAML config.
- `discover_urls.py` classification heuristics.

### Moderate-risk places
- `cleaning.py` pruning logic (can help or hurt dramatically).
- `branch_parser.py` fallback heuristics.
- retrieval filter logic in `vector_store.py`.

### High-risk places
- `models.py` schema fields (must keep compatibility across stages).
- `build_dataset.py` orchestration and dedupe logic.
- index write/load behavior in `vector_store.py`.

---

## 7) Mistakes that break the pipeline

- Changing schema fields without updating readers/writers/tests.
- Adding many irrelevant URLs to manifest (quality collapse due to noise).
- Over-aggressive selector removal (real content disappears).
- Switching embedding model without re-building index.
- Filtering retrieval by wrong bank identifier (name vs key confusion).
- Ignoring warnings in logs (`too_short`, `no sections`, `0 branch records`).

---

## 8) How to debug scraping problems

Use this sequence:

1. Check manifest URL health first (HTTP 200 and correct language path).
2. Open raw HTML in `data/raw_html/...` for failed page.
3. Inspect logs for:
   - fetch failure
   - low-value extraction warnings
   - no sections found
4. Compare raw HTML vs cleaned page in `data/cleaned_docs/...`.
5. If content is dynamic:
   - inspect network requests in browser DevTools
   - add endpoint logic to bank `fetch_structured()`
6. Add a regression test for that page pattern.

---

## 9) How to debug retrieval quality problems

Symptoms and causes:

- **Relevant answer not retrieved**
  - Missing URL in manifest
  - Poor chunking (topic mixed or section lost)
  - Cleaner removed key facts

- **Wrong bank/topic returned**
  - Filter not set or set incorrectly
  - Metadata mismatch in chunk records

- **Generic/noisy chunks dominate**
  - Too much navigation/footer text survived cleaning
  - Manifest includes broad non-topic pages

Debug workflow:

1. Start with one failed query.
2. Run `demo-retrieve` with and without filters.
3. Inspect top chunks and their source URLs.
4. Open source cleaned docs and raw HTML.
5. Fix manifest/cleaning/chunking in that order.
6. Re-run scrape + index; compare retrieval again.

---

## 10) How to evaluate if embeddings are working well

Use practical checks (not only cosine scores):

### A) Query set evaluation
- Prepare ~20 Armenian queries:
  - credits: rates, terms, collateral
  - deposits: rates, maturities, conditions
  - branches: address/hours/phone
- For each query, define expected bank/topic/URL pattern.

### B) Retrieval hit-rate checks
- Measure:
  - Top-1 relevant?
  - Top-3 contains relevant?
  - Correct bank/topic ranking?

### C) Embedding sanity checks
- Similar Armenian phrasings should retrieve similar chunks.
- Query in Armenian should outperform English for Armenian pages.
- If results are random/noisy, check chunk quality first (usually not the model itself).

### D) Comparative model check (optional)
- Swap model via config/env.
- Rebuild index.
- Re-run same query set.
- Keep model that improves Top-3 relevance while preserving speed.

---

## 11) Recommended owner workflow for changes

1. Update one small thing (manifest or rule).
2. Re-run scrape for one bank/topic only.
3. Inspect `raw_html`, `cleaned_docs`, `chunks`, `branches`.
4. Rebuild index.
5. Run retrieval smoke queries.
6. Add/update tests.
7. Only then expand scope.

This keeps changes explainable and interview-friendly.

---

## 12) Quick mental model for interview explanations

“This system is a deterministic offline ETL-to-retrieval pipeline:
manifest-constrained sources -> robust fetch -> bank-aware cleaning/extraction -> section-aware chunking -> embedding/indexing -> inspectable retrieval.  
The architecture is modular, test-backed, and optimized for quality of retrieved bank facts before adding live voice orchestration.”
