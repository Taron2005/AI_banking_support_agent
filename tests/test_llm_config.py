from pathlib import Path

from voice_ai_banking_support_agent.runtime.llm_config import load_llm_settings


def test_load_llm_settings_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_llm_settings(tmp_path / "missing.yaml")
    assert cfg.provider == "mock"
    assert cfg.model


def test_load_llm_settings_from_file(tmp_path: Path) -> None:
    p = tmp_path / "llm.yaml"
    p.write_text("provider: openai_compatible_http\nendpoint: http://127.0.0.1:8000/v1/chat/completions\n", encoding="utf-8")
    cfg = load_llm_settings(p)
    assert cfg.provider == "openai_compatible_http"
    assert cfg.endpoint is not None

