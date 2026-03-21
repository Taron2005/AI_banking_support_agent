"""Gemini client unit tests (mocked API; no real network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_ai_banking_support_agent.runtime.llm import GeminiChatClient


class _FakeResponse:
    def __init__(self, text: str = "Համառոտ պատասխան։") -> None:
        self.candidates = [object()]
        self.text = text


def test_gemini_generate_returns_text() -> None:
    client = GeminiChatClient(
        api_key="x",
        model="gemini-2.0-flash",
        timeout_seconds=30,
        temperature=0.1,
        max_output_tokens=256,
    )
    fake = _FakeResponse()
    with patch("google.generativeai.GenerativeModel") as m_cls:
        inst = MagicMock()
        inst.generate_content.return_value = fake
        m_cls.return_value = inst
        out = client.generate("prompt")
    assert "Համառոտ" in out
    inst.generate_content.assert_called_once()


def test_gemini_generate_raises_when_empty() -> None:
    client = GeminiChatClient(
        api_key="x",
        model="gemini-2.0-flash",
        timeout_seconds=30,
        temperature=0.1,
        max_output_tokens=256,
    )
    bad = _FakeResponse(text="   ")
    bad.text = "   "
    with patch("google.generativeai.GenerativeModel") as m_cls:
        inst = MagicMock()
        inst.generate_content.return_value = bad
        m_cls.return_value = inst
        with pytest.raises(RuntimeError, match="gemini_empty_response"):
            client.generate("p")


def test_gemini_missing_key() -> None:
    client = GeminiChatClient(
        api_key="",
        model="gemini-2.0-flash",
        timeout_seconds=30,
        temperature=0.1,
        max_output_tokens=256,
    )
    with pytest.raises(RuntimeError, match="gemini_missing_api_key"):
        client.generate("x")
