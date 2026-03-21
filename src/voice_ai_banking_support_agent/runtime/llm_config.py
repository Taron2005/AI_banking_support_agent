from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ProviderName = Literal["mock", "gemini"]

_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class LLMSettings(BaseModel):
    """LLM configuration. Production path: provider gemini + GEMINI_API_KEY."""

    provider: ProviderName = "gemini"
    endpoint: str | None = None  # unused for Gemini (reserved for future / evaluators)
    api_key: str | None = None
    model: str = _DEFAULT_GEMINI_MODEL
    timeout_seconds: float = 120.0
    temperature: float = 0.06
    max_tokens: int = 8192

    def resolved_api_key(self) -> str:
        """Effective API key after YAML + env (call after load_dotenv)."""

        ak = (self.api_key or "").strip()
        if ak:
            return ak
        if self.provider == "gemini":
            return (
                (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
            )
        return ""

    def is_live_llm_configured(self) -> bool:
        if self.provider == "mock":
            return True
        return bool(self.resolved_api_key())


_DEPRECATED_PROVIDERS = frozenset(
    {
        "groq",
        "openrouter",
        "cerebras",
        "ollama",
        "openai",
        "openai_compatible_http",
        "gemini_rest",
    }
)


def load_llm_settings(path: str | Path | None = None) -> LLMSettings:
    raw: dict = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    prov_raw = str(raw.get("provider", "")).strip().lower()
    if prov_raw in _DEPRECATED_PROVIDERS:
        logger.warning(
            "LLM provider %r is deprecated; this project uses Google Gemini only. Coercing to provider=gemini.",
            prov_raw or "(missing)",
        )
        raw["provider"] = "gemini"

    cfg = LLMSettings.model_validate(raw)

    env_prov = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_prov in _DEPRECATED_PROVIDERS:
        logger.warning("LLM_PROVIDER=%s is deprecated; use gemini or mock.", env_prov)
        cfg = cfg.model_copy(update={"provider": "gemini"})
    elif env_prov in ("mock", "gemini"):
        cfg = cfg.model_copy(update={"provider": env_prov})  # type: ignore[arg-type]

    # Explicit env keys for Gemini (do not read legacy GROQ_* as active keys)
    gem = (os.getenv("GEMINI_API_KEY") or "").strip()
    if gem:
        cfg = cfg.model_copy(update={"api_key": gem})
    ggl = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if ggl and not cfg.api_key:
        cfg = cfg.model_copy(update={"api_key": ggl})

    llm_key = (os.getenv("LLM_API_KEY") or "").strip()
    if llm_key and not cfg.api_key:
        cfg = cfg.model_copy(update={"api_key": llm_key})

    if (os.getenv("GROQ_API_KEY") or "").strip():
        logger.warning(
            "GROQ_API_KEY is set but Groq is not part of the active LLM path; it is ignored. Use GEMINI_API_KEY."
        )

    if os.getenv("LLM_MODEL"):
        cfg = cfg.model_copy(update={"model": os.getenv("LLM_MODEL", cfg.model)})

    mt = (os.getenv("LLM_MAX_TOKENS") or "").strip()
    if mt:
        try:
            cfg = cfg.model_copy(update={"max_tokens": int(mt)})
        except ValueError:
            pass

    return cfg
