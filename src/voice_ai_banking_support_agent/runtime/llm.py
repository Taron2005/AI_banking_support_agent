from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Iterator, Protocol

from .llm_config import LLMSettings
from .rag_prompts import RAG_ANSWER_SYSTEM_MESSAGE_EN

logger = logging.getLogger(__name__)

# English system instruction (canonical text in rag_prompts.RAG_ANSWER_SYSTEM_MESSAGE_EN).
RAG_SYSTEM_MESSAGE = RAG_ANSWER_SYSTEM_MESSAGE_EN


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


def _import_google_genai():
    """Lazy import so config/tests can load without google-generativeai until first LLM call."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as genai
        from google.api_core import exceptions as google_exceptions
    except ImportError as e:
        raise RuntimeError(
            "google-generativeai is not installed. Install project dependencies: pip install -r requirements.txt"
        ) from e
    return genai, google_exceptions


@dataclass
class MockLLMClient:
    """Deterministic local mock client for LLM integration tests."""

    prefix: str = "[MOCK-LLM]"

    def generate(self, prompt: str) -> str:
        text = prompt.replace("\n", " ")
        return f"{self.prefix} {text[:180]}..."

    def generate_stream(self, prompt: str) -> Iterator[str]:
        full = self.generate(prompt)
        for part in full.split():
            yield part + " "


@dataclass
class GeminiChatClient:
    """Google Gemini (Generative Language API) for grounded Armenian synthesis."""

    api_key: str
    model: str
    timeout_seconds: float
    temperature: float
    max_output_tokens: int
    system_message: str = RAG_SYSTEM_MESSAGE

    def generate(self, prompt: str) -> str:
        key = (self.api_key or "").strip()
        if not key:
            raise RuntimeError("gemini_missing_api_key")

        genai, google_exceptions = _import_google_genai()
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            self.model,
            system_instruction=self.system_message,
        )
        gcfg = genai.types.GenerationConfig(
            temperature=float(self.temperature),
            max_output_tokens=int(self.max_output_tokens),
        )

        to = int(self.timeout_seconds) if self.timeout_seconds else 120
        try:
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=gcfg,
                    request_options={"timeout": to},
                )
            except TypeError:
                # Older google-generativeai builds may not accept request_options here.
                response = model.generate_content(prompt, generation_config=gcfg)
        except google_exceptions.ResourceExhausted as e:
            logger.error("Gemini quota/rate limit: %s", e)
            raise RuntimeError("gemini_resource_exhausted") from e
        except google_exceptions.GoogleAPIError as e:
            logger.error("Gemini API error: %s", e)
            raise RuntimeError(f"gemini_api_error:{type(e).__name__}") from e
        except Exception as e:
            logger.exception("Gemini generate_content failed")
            raise RuntimeError(f"gemini_error:{type(e).__name__}") from e

        if not response.candidates:
            pf = getattr(response, "prompt_feedback", None)
            br = getattr(pf, "block_reason", None) if pf is not None else None
            logger.warning("Gemini returned no candidates block_reason=%s", br)
            raise RuntimeError(f"gemini_blocked:{br}")

        try:
            text = (response.text or "").strip()
        except Exception:
            text = ""
        if not text:
            fr = getattr(response.candidates[0], "finish_reason", None)
            logger.warning("Gemini returned empty text finish_reason=%s", fr)
            raise RuntimeError(f"gemini_empty_response:{fr}")

        return text

    def generate_stream(self, prompt: str) -> Iterator[str]:
        key = (self.api_key or "").strip()
        if not key:
            raise RuntimeError("gemini_missing_api_key")

        genai, google_exceptions = _import_google_genai()
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            self.model,
            system_instruction=self.system_message,
        )
        gcfg = genai.types.GenerationConfig(
            temperature=float(self.temperature),
            max_output_tokens=int(self.max_output_tokens),
        )
        to = int(self.timeout_seconds) if self.timeout_seconds else 120
        try:
            try:
                stream = model.generate_content(
                    prompt,
                    generation_config=gcfg,
                    stream=True,
                    request_options={"timeout": to},
                )
            except TypeError:
                stream = model.generate_content(prompt, generation_config=gcfg, stream=True)
        except google_exceptions.ResourceExhausted as e:
            logger.error("Gemini quota/rate limit: %s", e)
            raise RuntimeError("gemini_resource_exhausted") from e
        except google_exceptions.GoogleAPIError as e:
            logger.error("Gemini API error: %s", e)
            raise RuntimeError(f"gemini_api_error:{type(e).__name__}") from e
        except Exception as e:
            logger.exception("Gemini generate_content(stream) failed")
            raise RuntimeError(f"gemini_error:{type(e).__name__}") from e

        for chunk in stream:
            piece = _gemini_chunk_text(chunk)
            if piece:
                yield piece

        # Ensure stream is exhausted / surface empty final
        return


def _gemini_chunk_text(chunk: object) -> str:
    try:
        t = getattr(chunk, "text", None)
        if isinstance(t, str) and t:
            return t
    except Exception:
        pass
    try:
        cands = getattr(chunk, "candidates", None) or []
        if not cands:
            return ""
        parts = getattr(cands[0].content, "parts", None) or []
        out: list[str] = []
        for p in parts:
            txt = getattr(p, "text", None)
            if isinstance(txt, str) and txt:
                out.append(txt)
        return "".join(out)
    except Exception:
        return ""


def build_llm_client(settings: LLMSettings | None) -> LLMClient | None:
    if settings is None:
        return None
    if settings.provider == "mock":
        return MockLLMClient()

    if settings.provider == "gemini":
        key = settings.resolved_api_key()
        if not key:
            logger.error(
                "Gemini is configured (llm_config / LLM_PROVIDER) but no API key was found. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY (or LLM_API_KEY) in .env. "
                "Answers will use explicit extractive fallback until a key is set."
            )
            return None
        try:
            _import_google_genai()
        except RuntimeError as e:
            logger.error("%s", e)
            return None
        client = GeminiChatClient(
            api_key=key,
            model=(settings.model or "gemini-2.0-flash").strip(),
            timeout_seconds=float(settings.timeout_seconds),
            temperature=float(settings.temperature),
            max_output_tokens=int(settings.max_tokens),
            system_message=RAG_SYSTEM_MESSAGE,
        )
        logger.info(
            "LLM client ready: provider=gemini model=%s max_output_tokens=%s timeout_s=%s",
            client.model,
            client.max_output_tokens,
            client.timeout_seconds,
        )
        return client

    logger.warning("Unknown LLM provider %s; expected gemini or mock.", getattr(settings, "provider", None))
    return None
