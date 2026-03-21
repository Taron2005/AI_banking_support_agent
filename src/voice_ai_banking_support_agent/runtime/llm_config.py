from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ProviderName = Literal["mock", "groq"]


class LLMSettings(BaseModel):
    provider: ProviderName = "mock"
    endpoint: str | None = None
    api_key: str | None = None
    model: str = "llama-3.1-8b-instant"
    timeout_seconds: float = 20.0
    temperature: float = 0.1


def load_llm_settings(path: str | Path | None = None) -> LLMSettings:
    raw: dict = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    # Map deprecated provider strings in YAML to groq (evaluators may have old files).
    prov_raw = str(raw.get("provider", "")).strip().lower()
    if prov_raw in ("ollama", "openai_compatible_http", "gemini_rest", "openai"):
        logger.warning("LLM provider %r is no longer supported; use groq or mock.", prov_raw)
        raw["provider"] = "groq"
    cfg = LLMSettings.model_validate(raw)
    env_prov = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_prov in ("ollama", "openai_compatible_http", "gemini_rest", "openai"):
        logger.warning("LLM_PROVIDER=%s is deprecated; set provider to groq in llm_config.yaml.", env_prov)
        cfg = cfg.model_copy(update={"provider": "groq"})
    elif env_prov in ("mock", "groq"):
        cfg = cfg.model_copy(update={"provider": env_prov})  # type: ignore[arg-type]
    if os.getenv("LLM_ENDPOINT"):
        cfg = cfg.model_copy(update={"endpoint": os.getenv("LLM_ENDPOINT")})
    if os.getenv("LLM_API_KEY"):
        cfg = cfg.model_copy(update={"api_key": os.getenv("LLM_API_KEY")})
    if os.getenv("GROQ_API_KEY"):
        cfg = cfg.model_copy(update={"api_key": os.getenv("GROQ_API_KEY")})
    if os.getenv("LLM_MODEL"):
        cfg = cfg.model_copy(update={"model": os.getenv("LLM_MODEL", cfg.model)})
    return cfg
