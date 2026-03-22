from pathlib import Path

import pytest

from voice_ai_banking_support_agent.config import load_config


def test_config_loads_with_yaml_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRAPER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SCRAPER_RETRIES", raising=False)
    monkeypatch.delenv("SCRAPER_USER_AGENT", raising=False)
    monkeypatch.delenv("SCRAPER_REQUEST_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_NAME", raising=False)
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)
    monkeypatch.delenv("FAISS_USE_GPU", raising=False)
    monkeypatch.delenv("FAISS_GPU_ID", raising=False)
    project_root = tmp_path
    (project_root / "manifests").mkdir(parents=True, exist_ok=True)
    (project_root / "manifests" / "banks.yaml").write_text(
        'schema_version: "1"\nbanks: []\n', encoding="utf-8"
    )
    cfg_yaml = project_root / "config.yaml"
    cfg_yaml.write_text(
        "embedding_model_name: custom/model\n"
        "embedding_device: cuda\n"
        "embedding_batch_size: 16\n"
        "faiss_use_gpu: true\n"
        "faiss_gpu_id: 1\n"
        "network:\n  timeout_seconds: 12\n",
        encoding="utf-8",
    )
    cfg = load_config(project_root=project_root, config_yaml=cfg_yaml)
    assert cfg.embedding_model_name == "custom/model"
    assert cfg.embedding_device == "cuda"
    assert cfg.embedding_batch_size == 16
    assert cfg.faiss_use_gpu is True
    assert cfg.faiss_gpu_id == 1
    assert cfg.network.timeout_seconds == 12
