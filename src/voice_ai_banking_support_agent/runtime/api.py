from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

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
    # Note: load `.env` in the process entrypoint (e.g. run_runtime_api.py, voice CLI), not here,
    # so tests and library imports are not polluted by a developer's environment.

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
        return {
            "status": "ok",
            "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
            "livekit_api_configured": bool(
                os.getenv("LIVEKIT_API_KEY", "").strip() and os.getenv("LIVEKIT_API_SECRET", "").strip()
            ),
            "livekit_url": lk["livekit_url"],
            "livekit_room": lk["room"],
            "default_index": "hy_model_index",
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
        return resp.model_dump()

    return app
