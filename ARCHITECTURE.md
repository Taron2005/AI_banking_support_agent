# Architecture

## Scope

This repository implements the offline knowledge layer for an Armenian banking assistant:
- source control via manifest
- scraping + extraction
- chunking + embeddings
- local retrieval index

It intentionally excludes online answer generation and voice orchestration.

## Data Flow

1. `manifests/banks.yaml` defines allowed URLs by bank/topic.
2. `build_dataset` fetches pages and stores raw HTML in `data/raw_html/`.
3. Cleaning + section parsing creates page-level artifacts in `data/cleaned_docs/`.
4. Branch pages additionally produce structured records in `data/branches/`.
5. Section-aware chunking writes chunk JSONL files in `data/chunks/`.
6. `build_index` embeds chunk text and writes FAISS + metadata in `data/index/<index_name>/`.
7. `demo-retrieve` embeds queries and searches FAISS with optional topic/bank filters.

## Core Modules

- `config.py`: centralized runtime and path configuration.
- `bank_manifest.py`: strict manifest schema and URL validation.
- `scrapers/base.py`: HTTP fetcher with retries/session reuse and structured extraction helpers.
- `scrapers/*.py`: bank-specific extraction rules and `fetch_structured()` logic (including Ameriabank DNN module API payload extraction for JS-loaded pages).
- `extraction/*`: cleaning, section parsing, branch extraction.
- `indexing/*`: chunking, embedding, vector search.
- `pipelines/*`: orchestration for dataset, index build, URL discovery.

## Reliability and Failure Handling

- Per-page failures are logged and do not stop the run.
- Dataset run ends with failure ratio reporting and fails if too many pages break.
- JSONL outputs are de-duplicated by stable keys (chunk id or semantic key).
- Vector store validates metadata/vector count consistency at search time.

## Extensibility

To add a new bank:
1. add manifest entry with topic URLs,
2. add scraper file with extraction rules + `fetch_structured()`,
3. register scraper in `build_dataset.py`,
4. add bank-focused tests.

No pipeline rewrite is required.
