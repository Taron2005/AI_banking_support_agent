from pathlib import Path

import pytest

from voice_ai_banking_support_agent.runtime.llm import GeminiChatClient, build_llm_client
from voice_ai_banking_support_agent.runtime.llm_config import LLMSettings, load_llm_settings


def test_load_llm_settings_defaults_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = load_llm_settings(tmp_path / "missing.yaml")
    assert cfg.provider == "gemini"
    assert "gemini" in cfg.model.lower()


def test_groq_yaml_coerced_to_gemini(tmp_path: Path) -> None:
    p = tmp_path / "llm.yaml"
    p.write_text("provider: groq\nmodel: x\n", encoding="utf-8")
    cfg = load_llm_settings(p)
    assert cfg.provider == "gemini"


def test_deprecated_yaml_provider_coerced_to_gemini(tmp_path: Path) -> None:
    p = tmp_path / "llm.yaml"
    p.write_text("provider: openrouter\nmodel: x\n", encoding="utf-8")
    cfg = load_llm_settings(p)
    assert cfg.provider == "gemini"


def test_build_gemini_client_shape() -> None:
    client = build_llm_client(LLMSettings(provider="gemini", api_key="test-key", model="gemini-2.0-flash"))
    assert client is not None
    assert isinstance(client, GeminiChatClient)
    assert hasattr(client, "generate")


def test_build_gemini_returns_none_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert build_llm_client(LLMSettings(provider="gemini", api_key=None, model="gemini-2.0-flash")) is None


def test_build_gemini_returns_none_when_sdk_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_ai_banking_support_agent.runtime import llm as llm_mod

    def _boom():
        raise RuntimeError("google-generativeai is not installed")

    monkeypatch.setattr(llm_mod, "_import_google_genai", _boom)
    assert build_llm_client(LLMSettings(provider="gemini", api_key="test-key", model="gemini-2.0-flash")) is None


def test_gemini_api_key_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    p = tmp_path / "llm.yaml"
    p.write_text("provider: gemini\nmodel: gemini-2.0-flash\n", encoding="utf-8")
    cfg = load_llm_settings(p)
    assert cfg.resolved_api_key() == "gk-test"
    assert cfg.is_live_llm_configured()


def test_llm_model_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_MODEL", "gemini-1.5-pro")
    p = tmp_path / "llm.yaml"
    p.write_text("provider: gemini\nmodel: gemini-2.0-flash\n", encoding="utf-8")
    cfg = load_llm_settings(p)
    assert cfg.model == "gemini-1.5-pro"


def test_mock_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    p = tmp_path / "llm.yaml"
    p.write_text("provider: gemini\n", encoding="utf-8")
    cfg = load_llm_settings(p)
    assert cfg.provider == "mock"
