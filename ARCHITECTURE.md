# Architecture

## Scope

This repository is the full **Armenian banking voice + text assistant** stack: manifest-driven **ingestion** (scrape → clean → chunk → FAISS), **runtime RAG** (FastAPI `POST /chat`, topic/bank gating, Gemini or extractive fallback), and **voice** (LiveKit agent, HTTP STT/TTS). The **canonical high-level overview** is [README.md](README.md); this file summarizes module layout and the **offline data path** under `data_manifest_update_hy/` (configured in `validation_manifest_update_hy.yaml`).

## Data flow (ingestion)

1. `manifests/banks.yaml` defines allowed URLs by bank/topic.
2. **Scrape** fetches pages and stores raw HTML in `data_manifest_update_hy/raw_html/` (local cache; not committed).
3. Cleaning + section parsing creates artifacts in `data_manifest_update_hy/cleaned_docs/`.
4. Branch pages additionally produce structured records in `data_manifest_update_hy/branches/`.
5. Section-aware chunking writes chunk JSONL files in `data_manifest_update_hy/chunks/`.
6. **Build index** embeds chunk text and writes FAISS + metadata in `data_manifest_update_hy/index/<index_name>/`.
7. At query time, **retrieve** embeds the user query and searches FAISS with topic/bank filters (see `runtime/` and `indexing/`).

## Core modules (by area)

- **Config & manifest:** `config.py`, `bank_manifest.py`, YAML manifests under repo root and `manifests/`.
- **Scraping:** `scrapers/base.py`, `scrapers/*.py` — bank-specific extraction and `fetch_structured()`.
- **Extraction:** `extraction/*` — cleaning, section parsing, branch extraction.
- **Indexing:** `indexing/*` — chunking, embedding, vector search.
- **Pipelines:** `pipelines/*` — dataset build, index build, URL discovery.
- **Runtime API & RAG:** `runtime/*`, `run_runtime_api.py` — orchestration, LLM, prompts, refusals.
- **Voice:** `voice/*` — LiveKit agent, STT/TTS clients, optional in-process vs HTTP runtime (see `RUNTIME_ARCHITECTURE.md`, `LIVEKIT_INTEGRATION_ARCHITECTURE.md`).

## Reliability and failure handling

- Per-page failures are logged and do not stop the run.
- Dataset run ends with failure ratio reporting and fails if too many pages break.
- JSONL outputs are de-duplicated by stable keys (chunk id or semantic key).
- Vector store validates metadata/vector count consistency at search time.

## Extensibility

To add a new bank: extend `manifests/banks.yaml`, add a scraper with extraction rules, register it in the dataset pipeline, and re-run scrape + build-index. No fork of core retrieval or orchestration is required.
