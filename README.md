# Armenian Voice AI Banking Support Agent

Production-style local stack: **official bank website data** (3 Armenian banks) → **FAISS + Armenian embeddings** → **grounded runtime** (credits / deposits / branches only) → **Groq** answers (evidence-only) → **HTTP Armenian STT/TTS** + **self-hosted LiveKit** with **push-to-talk** (mic never always-on).

## 1. Project overview

- **Ingestion**: manifest-driven scraping from ACBA, Ameriabank, IDBank (`manifests/banks.yaml`).
- **Retrieval**: `Metric-AI/armenian-text-embeddings-2-large` + **FAISS** (`indexing/`).
- **Runtime**: follow-up merge → topic gate → bank filter → retrieval → evidence check → **Groq-grounded Armenian answers** with extractive fallback (`runtime/`).
- **Voice**: Python agent on **self-hosted LiveKit**; **push-to-talk** (`voice.ptt` start/end) → **HTTP STT** (Whisper-style, `language=hy`) → same runtime as text → **HTTP TTS** (Armenian, WAV).
- **API**: FastAPI `/chat` + LiveKit token/config (`run_runtime_api.py`).
- **UI**: React product UI + voice controls (`frontend-react/`).

## 2. Architecture (short)

```
User (text or audio)
  → [LiveKit + STT optional] → text query
  → RuntimeOrchestrator: normalize → follow-up resolve → topic classify → bank detect → retrieve (FAISS + filters) → evidence → Groq (evidence-only) or extractive fallback
  → [TTS optional] → audio
```

LiveKit Cloud URLs are **rejected** by `voice/voice_config.py` (self-hosted OSS only).

## 3. Data pipeline

1. `scrape` — fetch manifest URLs, clean HTML, chunk → JSONL under dataset dir.
2. `build-index` — embed chunks, write `faiss.index` + `metadata.jsonl`.

**Production dataset (submission default)** — see `DATASETS.md`:

| Item | Value |
|------|--------|
| Config YAML | `validation_manifest_update_hy.yaml` |
| Root folder | `data_manifest_update_hy/` |
| Index name | `hy_model_index` |

## 4. Runtime flow

1. **TopicClassifier** — only `credit`, `deposit`, `branch`; weak-only queries → **ambiguous** (refusal to clarify). Unsupported intents → refusal.
2. **RuntimeRetriever** — query embedding + FAISS search with **topic** and optional **bank** metadata filters.
3. **EvidenceChecker** — minimum similarity + branch/address heuristics when needed.
4. **LLMAnswerGenerator** (default when `answer.backend: llm`) — **Groq** only; user prompt lists numbered evidence blocks; system prompt in `runtime/llm.py` forbids facts outside evidence.
5. **GroundedAnswerGenerator** — deterministic extractive fallback if Groq is unavailable, declines, or errors.

## 5. Voice flow (default = real HTTP path)

1. **Docker LiveKit** — `docker compose up -d` (defaults: `devkey` / `secret`, UDP range in `docker-compose.yml`).
2. **JWTs** — `GET /api/livekit/token?identity=web-user-1` (React + `run_voice_agent` when `LIVEKIT_TOKEN` unset).
3. **STT** — set **`VOICE_STT_ENDPOINT`** to a Whisper-compatible **POST multipart** endpoint: form field **`file`** (WAV), **`language=hy`**, JSON response with transcript in **`text`** (or `transcription`). UTF-8 Armenian preserved.
4. **TTS** — set **`VOICE_TTS_ENDPOINT`** to a service accepting JSON **`{ text, language, voice, format }`** and returning **WAV** (raw or base64 in `audio_base64` / `audio`).
5. **Push-to-talk** — browser sends reliable data on topic **`voice.ptt`**: `{"type":"start"}` then publishes mic; `{"type":"end"}` then unpublishes. Agent buffers only while **start** is active.
6. **Transcript feedback (UI)** — after **Stop & send**, the agent runs STT, then publishes **`voice.transcript.final`** with the recognized text (UTF‑8 Armenian). The React UI shows it in the chat with an **STT** badge *before* the assistant answer. Server **`voice.state`** includes `listening`, `processing` (with `detail`: `transcribing` / `answering`), `speaking`, `idle`, `busy`, `error`.
7. **Mock / CI** — set **`VOICE_USE_MOCK=1`** or `stt.provider: mock` / `tts.provider: mock` in a copied `voice_config.yaml`.

### Quick path: run local STT + TTS on your PC

1. Install server extras (Whisper + Edge TTS + miniaudio + multipart):

   ```bash
   pip install -e ".[voice_local_servers]"
   ```

2. **Terminal A — STT** (first run downloads the Whisper model):

   ```bash
   python scripts/voice_http_stt_server.py
   ```

3. **Terminal B — TTS** (MP3 from Edge is converted to WAV with **miniaudio** — no ffmpeg):

   ```bash
   python scripts/voice_http_tts_server.py
   ```

4. In **`.env`**:

   ```env
   VOICE_STT_ENDPOINT=http://127.0.0.1:8088/transcribe
   VOICE_TTS_ENDPOINT=http://127.0.0.1:8089/synthesize
   ```

   Leave **`VOICE_STT_API_KEY`** / **`VOICE_TTS_API_KEY`** empty for these local servers.

5. Optional: in **`voice_config.yaml`** (or example copy), set `tts.voice_name` to **`hy-AM-AnahitNeural`** (or keep **`default`** — the sample TTS server maps `default` to that voice).

Windows: **`scripts\run_voice_stt_server.bat`** and **`scripts\run_voice_tts_server.bat`** run the same apps.

## 6. Model choices

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `Metric-AI/armenian-text-embeddings-2-large` | Armenian-centric retrieval quality. |
| Answers | **Groq** (`llama-3.1-8b-instant`) after evidence gate | Fast, fluent Armenian; still grounded on retrieved chunks only. |
| Fallback | Extractive snippets | Used when `GROQ_API_KEY` is missing or the API errors. |
| STT/TTS | **HTTP default** (`http_whisper` / `http_tts` in `voice_config.example.yaml`) | Set `VOICE_STT_ENDPOINT` / `VOICE_TTS_ENDPOINT`; mock only if endpoints missing or `VOICE_USE_MOCK=1`. |

Set **`GROQ_API_KEY`** in `.env`. With **`runtime_config.yaml`** `answer.backend: llm` and **`llm_config.yaml`** `provider: groq`, the runtime **always** uses Groq when the key is present; otherwise it falls back to extractive answers (still grounded on chunks).

## 7. Setup

**Prerequisites**: Python **3.10+**, Node **18+** (for React), optional self-hosted [LiveKit server](https://github.com/livekit/livekit).

```bash
cd "Voice AI banking support agent"
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e ".[dev]"
pip install -e ".[voice]"       # only for LiveKit Python SDK
```

Copy **`.env.example` → `.env`** and set **`GROQ_API_KEY`**, **`VOICE_STT_ENDPOINT`**, **`VOICE_TTS_ENDPOINT`** for the full Armenian voice demo. Templates: `.env.voice.example`, `.env.backend.example`, `.env.frontend.example`.

**Install embedding model** on first retrieval (downloads from Hugging Face).

### Helper scripts (Windows / Linux)

| Step | Windows | Linux / macOS |
|------|---------|----------------|
| Create venv + install | `scripts\setup_env.bat` | `bash scripts/setup_env.sh` |
| LiveKit (Docker) | `scripts\run_livekit.bat` | `bash scripts/run_livekit.sh` |
| Backend API | `scripts\run_backend.bat` | `bash scripts/run_backend.sh` |
| Frontend | `scripts\run_frontend.bat` | `bash scripts/run_frontend.sh` |
| Voice agent | `scripts\run_voice_agent.bat` (fetches JWT from API if `LIVEKIT_TOKEN` empty) | `bash scripts/run_voice_agent.sh` |
| **Full stack** | **`START_STACK.bat`** or **`START_FRESH.bat`** (clears :8000 / :5173 + `docker compose down` first) | `bash scripts/run_all.sh` |
| **Stop stack** | **`STOP_STACK.bat`** | — |

**Manual token (offline / no API):** `python scripts/generate_livekit_token.py --identity banking-support-agent`

## 7b. Local end-to-end run (recommended)

**Canonical paths:** `validation_manifest_update_hy.yaml`, `data_manifest_update_hy/`, index **`hy_model_index`**. Answers use **Groq** after the evidence gate (`runtime_config.yaml` + `llm_config.yaml`); set **`GROQ_API_KEY`** in `.env`.

1. Copy **`.env.example` → `.env`** and set **`GROQ_API_KEY`** (and optionally `LIVEKIT_*` if not using Docker defaults).
2. **Install:** `scripts\setup_env.bat` or `bash scripts/setup_env.sh`.
3. **Run everything:** **`START_STACK.bat`** (waits until `GET /api/livekit/config` works — avoids wrong app on :8000) or **`scripts\run_all.bat`** / **`bash scripts/run_all.sh`** (same stack: Docker LiveKit, API, voice agent, Vite).
4. Open **`http://127.0.0.1:5173`** — **Connect voice** → **Mic** (speak Armenian) → **Stop & send**; text chat uses **`hy_model_index`** automatically.
5. **Checks:** `GET /health`, **`GET /ready`**, **`GET /api/livekit/config`**.

**Run services individually:** `run_livekit` / `run_backend` / `run_frontend` / `run_voice_agent` as in the table. API loads `.env` via **`python run_runtime_api.py`**.

### Testing without speech servers

| Mode | Use |
|------|-----|
| **`VOICE_USE_MOCK=1`** | Forces mock STT/TTS (silent TTS; STT returns placeholder for binary audio). |
| **`voice-smoke-test`** | CLI smoke with mock providers. |

### Troubleshooting

- **`LIVEKIT_TOKEN` missing** — voice agent exits immediately; generate with `scripts/generate_livekit_token.py`.  
- **LiveKit won’t connect in the browser** (`could not establish pc connection`) — WebRTC/ICE issue. This repo maps UDP **50000-50050** for LiveKit (see `docker-compose.yml`). After editing `docker/livekit.yaml`, run `docker compose up -d --force-recreate`. Allow those UDP ports in Windows Firewall for Docker Desktop if needed. Set `LIVEKIT_URL=ws://127.0.0.1:7880` (not `http://`).  
- **`LiveKit config HTTP 404`** — port **8000** is serving a different app (or an old build). Stop it and run **`python run_runtime_api.py`** from this repo; confirm `GET http://127.0.0.1:8000/api/livekit/config` returns JSON with `livekit_url`.  
- **Docker: port 7880 in use** — stop other LiveKit or change host port in `docker-compose.yml`.  
- **First `/chat` is slow** — embedding model download / FAISS load on cold start.  
- **CORS** — API allows `*`; if the browser still blocks, check `VITE_API_BASE_URL` matches the API origin.  
- **`livekit-api` ImportError** on token script — run `pip install -e ".[voice]"`.

## 8. Running the API (default: production config)

From repo root — **no extra flags required**:

```bash
python run_runtime_api.py
```

Defaults:

- `--config validation_manifest_update_hy.yaml`
- `--runtime-config runtime_config.yaml`
- `--llm-config llm_config.yaml`
- `http://127.0.0.1:8000`

**POST** `/chat` JSON body:

```json
{
  "session_id": "eval-session-1",
  "query": "Ամերիաբանկում ինչ ավանդներ կան",
  "index_name": "hy_model_index",
  "top_k": 8,
  "verbose": true
}
```

## 9. Running the voice agent

```bash
python cli.py --config validation_manifest_update_hy.yaml voice-agent ^
  --index-name hy_model_index ^
  --runtime-config runtime_config.yaml ^
  --llm-config llm_config.yaml ^
  --voice-config voice_config.yaml
```

Copy **`voice_config.example.yaml` → `voice_config.yaml`** (the example is git-tracked; your copy is git-ignored).

**Required env**: `LIVEKIT_URL`, `LIVEKIT_TOKEN`.  
**Real speech**: `stt.provider: http_whisper`, `tts.provider: http_tts` in `voice_config.yaml` + **`VOICE_STT_ENDPOINT`** / **`VOICE_TTS_ENDPOINT`** in `.env`.

Smoke test (mock STT/TTS, text queries):

```bash
python cli.py --config validation_manifest_update_hy.yaml voice-smoke-test ^
  --index-name hy_model_index ^
  --runtime-config runtime_config.yaml ^
  --llm-config llm_config.yaml ^
  --voice-config voice_config.yaml
```
(Use `voice_config.example.yaml` if you have not created `voice_config.yaml` yet.)

## 10. Running the frontend

```bash
python run_runtime_api.py
cd frontend-react
npm install
npm run dev
```

Open the Vite URL (usually `http://127.0.0.1:5173`). Set `VITE_API_BASE_URL` if the API is not on `127.0.0.1:8000`.

Default UI index: **`hy_model_index`** (matches API defaults).

## 11. CLI: scrape & index (rebuild from source)

```bash
python -m voice_ai_banking_support_agent.cli scrape --banks acba ameriabank idbank --topics credit deposit branch --config validation_manifest_update_hy.yaml
python -m voice_ai_banking_support_agent.cli build-index --index-name hy_model_index --banks acba ameriabank idbank --topics credit deposit branch --config validation_manifest_update_hy.yaml
```

## 12. Guardrails

- **Topics**: only credit / deposit / branch; other → refusal or ambiguous.
- **Retrieval**: FAISS + metadata filters; not whole-corpus prompting.
- **Evidence**: low similarity → refusal (`insufficient_evidence`).
- **LLM** (if enabled): system prompt forbids facts outside provided evidence block; empty evidence → extractive fallback.

## 13. Groq setup

1. `runtime_config.yaml` — `answer.backend: llm` (default in this repo).
2. `llm_config.yaml` — `provider: groq`, `model: llama-3.1-8b-instant`.
3. `.env` — `GROQ_API_KEY=...` (from [Groq console](https://console.groq.com)).

If **`GROQ_API_KEY`** is missing, the runtime **falls back to extractive answers** (still from retrieved chunks only). Automated tests may use a temporary `llm_config` with `provider: mock` (see `tests/`).

## 14. Limitations

- Bank HTML changes can break scrapers; manifests must be updated.
- Branch pages may not expose every branch in static HTML.
- Voice is **push-to-talk** (no continuous streaming STT); transcript appears after each **Stop & send**.
- Mock STT on binary audio returns a placeholder unless HTTP STT is configured.
- Duplicate legacy dataset trees were removed from the submission layout; use paths in `DATASETS.md` only.

## 15. Further reading

- `ARCHITECTURE.md`, `RUNTIME_ARCHITECTURE.md`, `LIVEKIT_INTEGRATION_ARCHITECTURE.md`, `DATASETS.md`
- Optional / historical notes: `docs/archive/` (see `docs/archive/README.md`)
