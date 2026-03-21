# Dataset layout (submission)

## Production tree (evaluators should use this)

| Role | Path |
|------|------|
| Chunks, raw HTML, branches, indexes | `data_manifest_update_hy/` |
| FAISS index name | `hy_model_index` → `data_manifest_update_hy/index/hy_model_index/` |
| App config file | `validation_manifest_update_hy.yaml` |

This tree includes **all three banks** (ACBA, Ameriabank, IDBank) × **three topics** (credit, deposit, branch) chunk JSONL files, built with **`Metric-AI/armenian-text-embeddings-2-large`**.

## Rebuild index after chunker changes

If `indexing/chunker.py` or chunking config changes, regenerate chunks and rebuild the canonical index:

```bash
python -m voice_ai_banking_support_agent.cli scrape --banks acba ameriabank idbank --topics credit deposit branch --config validation_manifest_update_hy.yaml
python -m voice_ai_banking_support_agent.cli build-index --index-name hy_model_index --banks acba ameriabank idbank --topics credit deposit branch --config validation_manifest_update_hy.yaml
```

## Local LiveKit (Docker)

From repo root: `docker compose up -d` (see `docker-compose.yml`). Uses dev credentials **devkey** / **secret** and `ws://127.0.0.1:7880`. Mint JWTs with `python scripts/generate_livekit_token.py`.
