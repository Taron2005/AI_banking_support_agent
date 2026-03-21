from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class LLMSettings(BaseModel):
    provider: Literal["mock", "openai_compatible_http", "gemini_rest"] = "mock"
    endpoint: str | None = None
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 25.0
    temperature: float = 0.1


def load_llm_settings(path: str | Path | None = None) -> LLMSettings:
    raw: dict = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = LLMSettings.model_validate(raw)
    if os.getenv("LLM_PROVIDER"):
        cfg.provider = os.getenv("LLM_PROVIDER", cfg.provider)  # type: ignore[assignment]
    if os.getenv("LLM_ENDPOINT"):
        cfg.endpoint = os.getenv("LLM_ENDPOINT")
    if os.getenv("LLM_API_KEY"):
        cfg.api_key = os.getenv("LLM_API_KEY")
    if os.getenv("LLM_GEMINI_API_KEY") and cfg.provider == "gemini_rest":
        cfg.api_key = os.getenv("LLM_GEMINI_API_KEY")
    if os.getenv("LLM_MODEL"):
        cfg.model = os.getenv("LLM_MODEL", cfg.model)
    return cfg

