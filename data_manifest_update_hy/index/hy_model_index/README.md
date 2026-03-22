# FAISS index output (generated)

These files are **not committed** to keep the repository lean and avoid large binary churn:

- `faiss.index`
- `metadata.jsonl`
- `index_info.json`

**After cloning**, build once from the committed chunk JSONL (GPU/CPU; first run downloads the embedding model):

```bash
python -m voice_ai_banking_support_agent.cli --project-root . --config validation_manifest_update_hy.yaml build-index --index-name hy_model_index --banks acba ameriabank idbank --topics credit deposit branch
```

Chunk files live under `data_manifest_update_hy/chunks/`. See `README.md` and `DATASETS.md` for the full pipeline.
