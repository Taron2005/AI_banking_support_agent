from pathlib import Path

from voice_ai_banking_support_agent.config import load_config


def test_config_loads_with_yaml_override(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "manifests").mkdir(parents=True, exist_ok=True)
    (project_root / "manifests" / "banks.yaml").write_text(
        'schema_version: "1"\nbanks: []\n', encoding="utf-8"
    )
    cfg_yaml = project_root / "config.yaml"
    cfg_yaml.write_text(
        "embedding_model_name: custom/model\nnetwork:\n  timeout_seconds: 12\n",
        encoding="utf-8",
    )
    cfg = load_config(project_root=project_root, config_yaml=cfg_yaml)
    assert cfg.embedding_model_name == "custom/model"
    assert cfg.network.timeout_seconds == 12
