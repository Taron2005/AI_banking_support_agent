"""
Submission QA: exercise /chat for credit/deposit/branch/refusal/follow-up via TestClient.

Requires canonical FAISS index in workspace. Gemini optional (extractive fallback if no API key).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "data_manifest_update_hy/index/hy_model_index/faiss.index"

pytestmark = pytest.mark.slow


def _embedding_load_failed(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if not isinstance(exc, OSError):
        return False
    if getattr(exc, "winerror", None) == 1455:
        return True
    msg = str(exc).lower()
    return "paging file" in msg or "cannot allocate memory" in msg


def _warm_embedding_via_chat(client: TestClient) -> None:
    """First /chat that reaches retrieval loads SentenceTransformer; may fail on low VM (Windows 1455)."""
    try:
        # Must be in-scope so orchestrator calls the retriever (short queries like "." are refused before embed).
        client.post(
            "/chat",
            json={
                "session_id": "__warm_embed__",
                "query": "Ameriabank վարկ",
                "index_name": "hy_model_index",
                "top_k": 1,
                "verbose": False,
            },
        )
    except OSError as e:
        if _embedding_load_failed(e):
            pytest.skip(
                "Embedding model needs more RAM/virtual memory (Windows: System > About > "
                "Advanced system settings > Performance > Advanced > Virtual memory — increase page file, "
                "or close Docker/other heavy apps)."
            )
        raise
    except MemoryError:
        pytest.skip("Not enough memory to load embedding model for e2e /chat tests.")


@pytest.fixture(scope="session")
def chat_client() -> TestClient:
    pytest.importorskip("fastapi")
    if not INDEX_FILE.is_file():
        pytest.skip("Canonical HY index not present")
    from voice_ai_banking_support_agent.runtime.api import build_app

    app = build_app(
        project_root=str(ROOT),
        config_path=str(ROOT / "validation_manifest_update_hy.yaml"),
        runtime_config_path=str(ROOT / "runtime_config.yaml"),
        llm_config_path=str(ROOT / "llm_config.yaml"),
    )
    with TestClient(app) as client:
        _warm_embedding_via_chat(client)
        yield client


def _chat(client: TestClient, session: str, query: str) -> dict:
    r = client.post(
        "/chat",
        json={
            "session_id": session,
            "query": query,
            "index_name": "hy_model_index",
            "top_k": 8,
            "verbose": False,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_qa_credit_question(chat_client: TestClient) -> None:
    out = _chat(chat_client, "qa-credit-1", "Ամերիաբանկում ինչ սպառողական վարկեր կան")
    assert out["status"] in ("answered", "refused")
    assert out["answer_text"]
    if out["status"] == "answered":
        assert len(out["answer_text"]) > 20


def test_qa_deposit_question(chat_client: TestClient) -> None:
    out = _chat(chat_client, "qa-dep-1", "IDBank-ում ինչ ավանդներ կան")
    assert out["status"] in ("answered", "refused")
    assert out["answer_text"]


def test_qa_branch_question(chat_client: TestClient) -> None:
    out = _chat(chat_client, "qa-br-1", "ACBA մասնաճյուղ Երևանում որտե՞ղ կա, հասցեով ասա")
    assert out["status"] in ("answered", "refused")
    assert out["answer_text"]


def test_qa_refusal_out_of_scope(chat_client: TestClient) -> None:
    # Avoid city names (e.g. Yerevan) that weak-match the branch topic and yield "ambiguous".
    out = _chat(chat_client, "qa-ref-1", "What is the weather like today?")
    assert out["status"] == "refused"
    assert out["refusal_reason"] == "out_of_scope"
    assert "վարկ" in out["answer_text"] or "ավանդ" in out["answer_text"]


def test_qa_follow_up_same_session(chat_client: TestClient) -> None:
    sid = "qa-follow-1"
    first = _chat(chat_client, sid, "Ինչ ավանդներ ունի Ամերիաբանկը")
    assert first["status"] in ("answered", "refused")
    second = _chat(chat_client, sid, "իսկ տոկոսադրույքը?")
    assert second["status"] in ("answered", "refused")
    assert second["answer_text"]


def test_qa_general_deposit_no_bank_named(chat_client: TestClient) -> None:
    """General question should not be silently scoped to one bank."""
    out = _chat(chat_client, "qa-gen-dep-1", "Ինչ ավանդային տարբերակներ կան բանկերում")
    # With orchestration.require_explicit_bank, a generic in-scope question may return clarify
    # until the user names a bank or asks for all banks — not a silent single-bank answer.
    assert out["status"] in ("answered", "refused", "clarify")
    assert out["answer_text"]
    if out["status"] == "answered":
        assert out.get("detected_bank") in (None, "")
        assert not out.get("detected_banks")
    if out["status"] == "clarify":
        assert len(out["answer_text"]) > 10


def test_qa_two_named_banks_returns_detected_banks(chat_client: TestClient) -> None:
    out = _chat(chat_client, "qa-cmp-1", "ACBA և Ameriabank ավանդների մասին համեմատի")
    assert out["status"] in ("answered", "refused")
    if out["status"] == "answered":
        assert set(out.get("detected_banks") or []) >= {"acba", "ameriabank"}
        assert out.get("detected_bank") in (None, "")
