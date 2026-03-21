# Demo Quickstart

## 1) Install

```bash
pip install -r requirements.txt
pip install -e ".[dev,voice]"
```

## 2) Use demo-ready config

- App config: `validation_manifest_update_hy.yaml`
- Runtime config: `runtime_config.yaml`
- Voice config: `voice_config.example.yaml`
- LLM config: `llm_config.yaml` (Gemini + `GEMINI_API_KEY`, or `provider: mock`)
- Index name: `hy_model_index`

## 3) Text frontend demo (fastest)

```bash
python frontend_demo.py --config validation_manifest_update_hy.yaml --runtime-config runtime_config.yaml --llm-config llm_config.yaml --index-name hy_model_index --port 8080
```

Open `http://127.0.0.1:8080`

## 3b) React frontend demo

Start backend API:

```bash
python run_runtime_api.py
```

In a second terminal:

```bash
cd frontend-react
npm install
npm run dev
```

Open the printed Vite URL (usually `http://127.0.0.1:5173`).

## 4) Voice smoke demo (STT->runtime->TTS chain)

```bash
python cli.py --config validation_manifest_update_hy.yaml voice-smoke-test --index-name hy_model_index --runtime-config runtime_config.yaml --llm-config llm_config.yaml --voice-config voice_config.example.yaml
```

## 5) Self-hosted LiveKit demo

1. Start LiveKit server: `livekit-server --dev`
2. Generate keys: `livekit-server generate-keys`
3. Generate token:
   `lk token create --api-key <KEY> --api-secret <SECRET> --join --room banking-support-room --identity banking-support-agent`
4. Export env:
   - `LIVEKIT_URL=ws://127.0.0.1:7880`
   - `LIVEKIT_API_KEY=<KEY>`
   - `LIVEKIT_API_SECRET=<SECRET>`
   - `LIVEKIT_TOKEN=<TOKEN>`
5. Run:
```bash
python cli.py --config validation_manifest_update_hy.yaml voice-agent --index-name hy_model_index --runtime-config runtime_config.yaml --llm-config llm_config.yaml --voice-config voice_config.example.yaml
```
