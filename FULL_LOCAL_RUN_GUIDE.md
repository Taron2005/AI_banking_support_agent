# Full Local Run Guide

## A. From clean clone

1. Create venv and activate.
2. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install -e ".[dev,voice]"`

## B. Data/index preparation

If indexes already exist (`data_manifest_update_multi/index/multi_model_index`), skip build.
Otherwise:

```bash
python cli.py --config demo_config.yaml scrape --banks acba ameriabank idbank --topics credit deposit branch
python cli.py --config demo_config.yaml build-index --index-name multi_model_index --banks acba ameriabank idbank --topics credit deposit branch
```

## C. Runtime test

```bash
python cli.py --config demo_config.yaml runtime-eval --index-name multi_model_index --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml
```

## D. Frontend test

```bash
python frontend_demo.py --config demo_config.yaml --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --index-name multi_model_index --port 8080
```

## E. Voice smoke test

```bash
python cli.py --config demo_config.yaml voice-smoke-test --index-name multi_model_index --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --voice-config voice_config.example.yaml
```

## F. Self-hosted LiveKit voice run

1. `livekit-server --dev`
2. `livekit-server generate-keys`
3. `lk token create --api-key <KEY> --api-secret <SECRET> --join --room banking-support-room --identity banking-support-agent`
4. Set env vars and run `voice-agent`.

## G. Real STT/TTS and LLM wiring

Edit:
- `voice_config.example.yaml` (`http_whisper`, `http_tts` endpoints)
- `llm_config.example.yaml` (`openai_compatible_http` endpoint/model/key)

If providers fail:
- voice providers can fallback to mock (config flags),
- LLM backend falls back to extractive output.
