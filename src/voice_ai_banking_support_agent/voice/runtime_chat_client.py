from __future__ import annotations

import logging
from typing import Any

import requests

from ..runtime.models import RuntimeResponse
from ..runtime.orchestrator import RuntimeRequest

logger = logging.getLogger(__name__)

_RUNTIME_KEYS = frozenset(RuntimeResponse.model_fields.keys())


def runtime_response_from_chat_payload(body: dict[str, Any]) -> RuntimeResponse:
    """Drop API-only keys (e.g. llm_provider) before building RuntimeResponse."""
    core = {k: body[k] for k in _RUNTIME_KEYS if k in body}
    return RuntimeResponse.model_validate(core)


class RuntimeChatClient:
    """
    Forwards each voice turn to the same FastAPI ``POST /chat`` handler the browser uses,
    so session state (pending clarify, follow-ups) is shared with text chat.
    """

    def __init__(self, base_url: str, *, timeout_seconds: float = 180.0) -> None:
        self._base = (base_url or "").strip().rstrip("/")
        self._timeout = float(timeout_seconds)

    @property
    def base_url(self) -> str:
        return self._base

    def chat(self, req: RuntimeRequest) -> RuntimeResponse:
        if not self._base:
            raise RuntimeError("RuntimeChatClient: empty base URL (set VOICE_RUNTIME_API_URL).")
        url = f"{self._base}/chat"
        payload = {
            "session_id": req.session_id,
            "query": req.query,
            "index_name": req.index_name,
            "top_k": req.top_k,
            "verbose": req.verbose,
        }
        try:
            r = requests.post(url, json=payload, timeout=self._timeout)
            r.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("POST %s failed", url)
            raise RuntimeError(
                f"Voice runtime API unreachable at {url}. Start the FastAPI server "
                f"(python run_runtime_api.py) or set VOICE_RUNTIME_HTTP=0 for in-process mode."
            ) from exc
        body = r.json()
        return runtime_response_from_chat_payload(body)
