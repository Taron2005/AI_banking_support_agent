from pathlib import Path

from voice_ai_banking_support_agent.config import load_config


def test_config_yaml_string_paths_are_converted_to_path_objects(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "manifests").mkdir(parents=True, exist_ok=True)
    (project_root / "manifests" / "banks.yaml").write_text(
        'schema_version: "1"\nbanks: []\n', encoding="utf-8"
    )
    cfg_yaml = project_root / "config.yaml"
    cfg_yaml.write_text(
        "\n".join(
            [
                "data_dir: data_validation",
                "raw_html_dir: data_validation/raw_html",
                "branches_dir: data_validation/branches",
                "cleaned_docs_dir: data_validation/cleaned_docs",
                "chunks_dir: data_validation/chunks",
                "index_dir: data_validation/index",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(project_root=project_root, config_yaml=cfg_yaml)
    assert isinstance(cfg.data_dir, Path)
    assert cfg.data_dir.is_absolute()
    assert cfg.data_dir.name == "data_validation"
