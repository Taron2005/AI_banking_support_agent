# Full Local Run Guide

## A. From clean clone

1. Create venv and activate.
2. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install -e ".[dev,voice]"`

## B. Data/index preparation

If indexes already exist (`data_manifest_update_hy/index/hy_model_index`), skip build.
Otherwise:

```bash
python cli.py --config validation_manifest_update_hy.yaml scrape --banks acba ameriabank idbank --topics credit deposit branch
python cli.py --config validation_manifest_update_hy.yaml build-index --index-name hy_model_index --banks acba ameriabank idbank --topics credit deposit branch
```

## C. Runtime test

```bash
python cli.py --config validation_manifest_update_hy.yaml runtime-eval --index-name hy_model_index --runtime-config runtime_config.yaml --llm-config llm_config.yaml
```

## D. Frontend test

```bash
python frontend_demo.py --config validation_manifest_update_hy.yaml --runtime-config runtime_config.yaml --llm-config llm_config.yaml --index-name hy_model_index --port 8080
```

## E. Voice smoke test

```bash
python cli.py --config validation_manifest_update_hy.yaml voice-smoke-test --index-name hy_model_index --runtime-config runtime_config.yaml --llm-config llm_config.yaml --voice-config voice_config.example.yaml
```

## F. Self-hosted LiveKit voice run

1. `livekit-server --dev`
2. `livekit-server generate-keys`
3. `lk token create --api-key <KEY> --api-secret <SECRET> --join --room banking-support-room --identity banking-support-agent`
4. Set env vars and run `voice-agent`.

## G. Real STT/TTS and Groq

Edit:
- `voice_config.example.yaml` (`http_whisper`, `http_tts` endpoints)
- `llm_config.yaml` — `provider: groq`, set `GROQ_API_KEY` in `.env`

If providers fail:
- voice providers can fallback to mock (config flags),
- Groq errors fall back to extractive answers.
