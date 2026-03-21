# Armenian Voice AI Banking Support Agent

End-to-end demo: **official bank website data** (3 Armenian banks) → **chunked embeddings (FAISS)** → **rules-first runtime** (credits / deposits / branches only) → optional **HTTP STT/TTS** + **self-hosted LiveKit** voice transport.

## 1. Project overview

- **Ingestion**: manifest-driven scraping from ACBA, Ameriabank, IDBank (`manifests/banks.yaml`).
- **Retrieval**: `Metric-AI/armenian-text-embeddings-2-large` + **FAISS** (`indexing/`).
- **Runtime**: follow-up merge → topic gate → bank filter → retrieval → evidence check → **Groq-grounded Armenian answers** with extractive fallback (`runtime/`).
- **Voice**: Python agent joins a **self-hosted LiveKit** room, runs STT → runtime → TTS (`voice/`).
- **API**: FastAPI `/chat` for text QA (`run_runtime_api.py`).
- **UI**: React chat + optional LiveKit mic (`frontend-react/`).

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

## 5. Voice flow

1. Set `LIVEKIT_URL`, `LIVEKIT_TOKEN` (and generate token with your server keys).
2. Optional: `VOICE_STT_ENDPOINT`, `VOICE_TTS_API_KEY`, etc. for real speech (`voice_config.example.yaml`).
3. Run `voice-agent` (see below). Mock STT returns a placeholder for raw audio unless HTTP STT is configured.

## 6. Model choices

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `Metric-AI/armenian-text-embeddings-2-large` | Armenian-centric retrieval quality. |
| Answers | **Groq** (`llama-3.1-8b-instant`) after evidence gate | Fast, fluent Armenian; still grounded on retrieved chunks only. |
| Fallback | Extractive snippets | Used when `GROQ_API_KEY` is missing or the API errors. |
| STT/TTS | Pluggable HTTP | Bring your Whisper / Piper / Coqui endpoint. |

Set **`GROQ_API_KEY`** in `.env` for fluent answers; without it, the stack falls back to extractive text automatically.

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

Copy `.env.example` → `.env` and adjust (optional). Split templates: `.env.backend.example`, `.env.voice.example`, `.env.frontend.example`.

**Install embedding model** on first retrieval (downloads from Hugging Face).

### Helper scripts (Windows / Linux)

| Step | Windows | Linux / macOS |
|------|---------|----------------|
| Create venv + install | `scripts\setup_env.bat` | `bash scripts/setup_env.sh` |
| LiveKit (Docker) | `scripts\run_livekit.bat` | `bash scripts/run_livekit.sh` |
| Backend API | `scripts\run_backend.bat` | `bash scripts/run_backend.sh` |
| Frontend | `scripts\run_frontend.bat` | `bash scripts/run_frontend.sh` |
| Voice agent | `scripts\run_voice_agent.bat` (needs `LIVEKIT_TOKEN`) | `bash scripts/run_voice_agent.sh` |

Token (requires `pip install -e ".[voice]"` for `livekit-api`):

```bash
python scripts/generate_livekit_token.py --identity banking-support-agent
python scripts/generate_livekit_token.py --identity web-user-1
```

## 7b. Local end-to-end run (ordered checklist)

**Canonical paths**: `validation_manifest_update_hy.yaml`, `data_manifest_update_hy/`, index `hy_model_index`, **`llm_config.yaml`** + **`runtime_config.yaml`** (`answer.backend: llm`).

1. **Clone / cd** into the repository root.
2. **Python venv**  
   - Windows: `scripts\setup_env.bat`  
   - Linux/macOS: `bash scripts/setup_env.sh`  
   Or manually: `python -m venv .venv`, activate, `pip install -r requirements.txt`, `pip install -e ".[dev,voice]"`.
3. **Environment**  
   - Copy `.env.example` → `.env`.  
   - Frontend (optional): `frontend-react/.env.example` → `.env.local`.
4. **LiveKit (Docker)**  
   - `docker compose up -d` (or `scripts\run_livekit.bat` / `run_livekit.sh`).  
   - Dev keys: `LIVEKIT_API_KEY=devkey`, `LIVEKIT_API_SECRET=secret`, `LIVEKIT_URL=ws://127.0.0.1:7880`.
5. **Generate JWTs**  
   - Agent: `python scripts/generate_livekit_token.py --identity banking-support-agent` → set `LIVEKIT_TOKEN` in `.env`.  
   - Browser (React LiveKit panel): generate a second token with `--identity web-user-1` and paste into the UI.
6. **Backend**  
   - `python run_runtime_api.py` (loads `.env` via `python-dotenv`).  
   - Check `http://127.0.0.1:8000/health` → `{"status":"ok",...}` and `/docs` for OpenAPI.
7. **Frontend**  
   - `cd frontend-react && npm install && npm run dev` — default index `hy_model_index`, API from `VITE_API_BASE_URL` or `http://127.0.0.1:8000`.
8. **Voice agent**  
   - With `LIVEKIT_TOKEN` set: `scripts\run_voice_agent.bat` or the `python cli.py ... voice-agent` command under §9.  
   - **Mock mode** (default `voice_config.example.yaml`): text and smoke tests work; **raw mic audio** without HTTP STT yields a placeholder transcript unless you configure `http_whisper`.
9. **Groq**  
   - `llm_config.yaml` → `provider: groq` and `GROQ_API_KEY` in `.env` (see `.env.example`).

### Mock vs real voice

| Mode | What works |
|------|------------|
| **Mock STT/TTS** | Voice smoke (`voice-smoke-test`), data-packet text path, valid silent mock WAV for TTS. |
| **Real STT/TTS** | Set `VOICE_STT_ENDPOINT` / `VOICE_TTS_ENDPOINT` and `provider: http_whisper` / `http_tts` in a copied `voice_config.yaml`. |

### Troubleshooting

- **`LIVEKIT_TOKEN` missing** — voice agent exits immediately; generate with `scripts/generate_livekit_token.py`.  
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
  --voice-config voice_config.example.yaml
```

**Required env**: `LIVEKIT_URL`, `LIVEKIT_TOKEN`.  
**Real speech**: set `stt.provider: http_whisper`, `tts.provider: http_tts` in a copied `voice_config.yaml` + `VOICE_STT_ENDPOINT` / `VOICE_TTS_ENDPOINT`.

Smoke test (mock STT/TTS, text queries):

```bash
python cli.py --config validation_manifest_update_hy.yaml voice-smoke-test ^
  --index-name hy_model_index ^
  --runtime-config runtime_config.yaml ^
  --llm-config llm_config.yaml ^
  --voice-config voice_config.example.yaml
```

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

For offline tests, set `provider: mock` in `llm_config.yaml` or omit `GROQ_API_KEY` (answers fall back to extractive).

## 14. Limitations

- Bank HTML changes can break scrapers; manifests must be updated.
- Branch pages may not expose every branch in static HTML.
- Voice uses ~3 s audio windows (no full VAD rewrite).
- Mock STT on binary audio returns a placeholder unless HTTP STT is configured.
- Duplicate legacy dataset trees were removed from the submission layout; use paths in `DATASETS.md` only.

## 15. Further reading

- `ARCHITECTURE.md`, `RUNTIME_ARCHITECTURE.md`, `LIVEKIT_INTEGRATION_ARCHITECTURE.md`
- `OWNER_GUIDE.md`, `DATASETS.md`, `DEMO_QUICKSTART.md`, `FULL_LOCAL_RUN_GUIDE.md`
