from pathlib import Path

import pytest

from voice_ai_banking_support_agent.runtime.llm import GroqChatClient, build_llm_client
from voice_ai_banking_support_agent.runtime.llm_config import LLMSettings, load_llm_settings


def test_load_llm_settings_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_llm_settings(tmp_path / "missing.yaml")
    assert cfg.provider == "mock"
    assert cfg.model


def test_load_llm_settings_groq_from_file(tmp_path: Path) -> None:
    p = tmp_path / "llm.yaml"
    p.write_text("provider: groq\nmodel: llama-3.1-8b-instant\n", encoding="utf-8")
    cfg = load_llm_settings(p)
    assert cfg.provider == "groq"
    assert "llama" in cfg.model


def test_deprecated_yaml_provider_coerced_to_groq(tmp_path: Path) -> None:
    p = tmp_path / "llm.yaml"
    p.write_text("provider: ollama\nmodel: llama3\n", encoding="utf-8")
    cfg = load_llm_settings(p)
    assert cfg.provider == "groq"


def test_build_llm_groq_client_shape() -> None:
    client = build_llm_client(
        LLMSettings(provider="groq", api_key="test", model="llama-3.1-8b-instant")
    )
    assert client is not None
    assert isinstance(client, GroqChatClient)
    assert hasattr(client, "generate")


def test_legacy_llm_provider_env_maps_to_groq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    p = tmp_path / "llm.yaml"
    p.write_text("provider: mock\n", encoding="utf-8")
    cfg = load_llm_settings(p)
    assert cfg.provider == "groq"
