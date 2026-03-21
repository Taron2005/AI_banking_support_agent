# Demo Quickstart

## 1) Install

```bash
pip install -r requirements.txt
pip install -e ".[dev,voice]"
```

## 2) Use demo-ready config

- App config: `demo_config.yaml`
- Runtime config: `runtime_config.yaml`
- Voice config: `voice_config.example.yaml`
- LLM config: `llm_config.example.yaml`

## 3) Text frontend demo (fastest)

```bash
python frontend_demo.py --config demo_config.yaml --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --index-name multi_model_index --port 8080
```

Open `http://127.0.0.1:8080`

## 3b) React frontend demo

Start backend API:

```bash
python run_runtime_api.py --config demo_config.yaml --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --host 127.0.0.1 --port 8000
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
python cli.py --config demo_config.yaml voice-smoke-test --index-name multi_model_index --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --voice-config voice_config.example.yaml
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
python cli.py --config demo_config.yaml voice-agent --index-name multi_model_index --runtime-config runtime_config.yaml --llm-config llm_config.example.yaml --voice-config voice_config.example.yaml
```
