from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[1].joinpath("data_manifest_update_hy/index/hy_model_index/faiss.index").is_file(),
    reason="HY index not present in workspace",
)
def test_health_and_root_with_full_app() -> None:
    root = Path(__file__).resolve().parents[1]
    from voice_ai_banking_support_agent.runtime.api import build_app

    app = build_app(
        project_root=str(root),
        config_path=str(root / "validation_manifest_update_hy.yaml"),
        runtime_config_path=str(root / "runtime_config.yaml"),
        llm_config_path=str(root / "llm_config.yaml"),
    )
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
    r2 = client.get("/")
    assert r2.status_code == 200
    assert "docs" in r2.json()
