from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ..config import load_config
from .factory import build_runtime_orchestrator
from .llm_config import load_llm_settings
from .orchestrator import RuntimeRequest
from .runtime_config import load_runtime_settings
from .session_state import SessionStateStore


class ChatRequest(BaseModel):
    session_id: str
    query: str
    index_name: str
    top_k: int = 6
    verbose: bool = False


def build_app(
    project_root: str = ".",
    config_path: str | None = None,
    runtime_config_path: str | None = None,
    llm_config_path: str | None = None,
):
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Install `fastapi` and `uvicorn`.") from e

    cfg = load_config(
        project_root=Path(project_root).resolve(),
        config_yaml=Path(config_path).resolve() if config_path else None,
    )
    runtime_settings = load_runtime_settings(runtime_config_path)
    llm_settings = load_llm_settings(llm_config_path)
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

