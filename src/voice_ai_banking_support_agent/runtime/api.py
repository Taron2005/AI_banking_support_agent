from __future__ import annotations

import logging
import os
from pathlib import Path

# Before transformers / sentence_transformers (via retriever → embedder). run_runtime_api.py also sets these.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from ..config import load_config
from .factory import build_runtime_orchestrator
from .livekit_tokens import livekit_env_config, mint_participant_token
from .llm_config import load_llm_settings
from .orchestrator import RuntimeRequest
from .runtime_config import load_runtime_settings
from .session_state import SessionStateStore


class ChatRequest(BaseModel):
    session_id: str
    query: str
    index_name: str = "hy_model_index"
    top_k: int = 8
    verbose: bool = False


class LiveKitTokenBody(BaseModel):
    identity: str = Field(..., min_length=1, max_length=128)
    room: str | None = None


def build_app(
    project_root: str = ".",
    config_path: str | None = None,
    runtime_config_path: str | None = None,
    llm_config_path: str | None = None,
):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Install `fastapi` and `uvicorn`.") from e

    root = Path(project_root).resolve()
    # Load repo-root `.env` before LLM settings so GEMINI_API_KEY is visible regardless of entrypoint
    # (uvicorn module path, IDE "Run", tests with TestClient). `override=False` keeps real env vars.
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    cfg = load_config(
        project_root=root,
        config_yaml=Path(config_path).resolve() if config_path else None,
    )
    runtime_settings = load_runtime_settings(runtime_config_path)
    if llm_config_path:
        llm_path: Path | None = Path(llm_config_path).resolve()
    else:
        default_llm = root / "llm_config.yaml"
        llm_path = default_llm if default_llm.exists() else None
    llm_settings = load_llm_settings(llm_path)
    orchestrator = build_runtime_orchestrator(
        app_config=cfg, runtime_settings=runtime_settings, llm_settings=llm_settings
    )
    _llm_ok = llm_settings.is_live_llm_configured()
    logger.info(
        "Runtime ready: answer.backend=%s llm_provider=%s llm_model=%s llm_configured=%s index_dir=%s",
        runtime_settings.answer.backend,
        llm_settings.provider,
        llm_settings.model,
        _llm_ok,
        cfg.index_dir,
    )
    if runtime_settings.answer.backend == "llm" and llm_settings.provider == "gemini" and not _llm_ok:
        logger.error(
            "answer.backend is llm but Gemini is not configured (set GEMINI_API_KEY or GOOGLE_API_KEY in .env). "
            "Every answered turn will use explicit extractive fallback until a key is set."
        )
    sessions = SessionStateStore()
    app = FastAPI(title="Voice AI Banking Runtime API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "voice-ai-banking-runtime"}

    @app.get("/ready")
    def ready():
        lk = livekit_env_config()
        llm_cfg_path = str(llm_path.resolve()) if llm_path else None
        return {
            "status": "ok",
            "llm_configured": _llm_ok,
            "livekit_api_configured": bool(
                os.getenv("LIVEKIT_API_KEY", "").strip() and os.getenv("LIVEKIT_API_SECRET", "").strip()
            ),
            "livekit_url": lk["livekit_url"],
            "livekit_room": lk["room"],
            "default_index": "hy_model_index",
            "answer_backend": runtime_settings.answer.backend,
            "llm_provider": llm_settings.provider,
            "llm_model": llm_settings.model,
            "llm_config_path": llm_cfg_path,
            "index_dir": str(cfg.index_dir.resolve()),
            "embedding_model_name": cfg.embedding_model_name,
        }

    @app.get("/")
    def root():
        return {"service": "Voice AI Banking Runtime API", "docs": "/docs", "health": "/health", "ready": "/ready"}

    @app.get("/api/livekit/config")
    def livekit_config_public():
        return livekit_env_config()

    @app.get("/api/livekit/token")
    def livekit_token_get(
        identity: str = Query(..., min_length=1, max_length=128),
        room: str | None = Query(None),
    ):
        try:
            tok = mint_participant_token(identity=identity, room=room)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"token": tok, **livekit_env_config()}

    @app.post("/api/livekit/token")
    def livekit_token_post(body: LiveKitTokenBody):
        try:
            tok = mint_participant_token(identity=body.identity, room=body.room)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"token": tok, **livekit_env_config()}

    @app.post("/chat")
    def chat(req: ChatRequest):
        state = sessions.get_or_create(req.session_id)
        resp = orchestrator.handle(
            RuntimeRequest(
                session_id=req.session_id,
                query=req.query,
                index_name=req.index_name,
                top_k=req.top_k,
                verbose=req.verbose,
            ),
            state,
        )
        payload = resp.model_dump()
        payload["llm_provider"] = llm_settings.provider
        payload["llm_model"] = llm_settings.model
        return payload

    return app
