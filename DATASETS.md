# Dataset layout (submission)

## Production tree (evaluators should use this)

| Role | Path |
|------|------|
| Chunks, cleaned docs, branches, indexes | `data_manifest_update_hy/` |
| Raw HTML cache (local only, gitignored) | `data_manifest_update_hy/raw_html/` — produced by **`scrape`**, omitted from the remote repo to keep size down |
| FAISS index name | `hy_model_index` → `data_manifest_update_hy/index/hy_model_index/` |
| App config file | `validation_manifest_update_hy.yaml` |

This tree includes **all three banks** (ACBA, Ameriabank, IDBank) × **three topics** (credit, deposit, branch) chunk JSONL files, built with **`Metric-AI/armenian-text-embeddings-2-large`**.

## Rebuild index after chunker changes

If `indexing/chunker.py` or chunking config changes, regenerate chunks and rebuild the canonical index:

```bash
python cli.py --config validation_manifest_update_hy.yaml scrape --banks acba ameriabank idbank --topics credit deposit branch
python cli.py --config validation_manifest_update_hy.yaml build-index --index-name hy_model_index --banks acba ameriabank idbank --topics credit deposit branch
```

## Local LiveKit (Docker)

From repo root: `docker compose up -d` (see `docker-compose.yml`). Uses dev credentials **devkey** / **secret** and `ws://127.0.0.1:7880`. Mint JWTs with `python scripts/generate_livekit_token.py`.
