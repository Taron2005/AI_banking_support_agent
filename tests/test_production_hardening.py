from pathlib import Path

import numpy as np
import pytest

from voice_ai_banking_support_agent.bank_manifest import load_banks_manifest
from voice_ai_banking_support_agent.indexing.vector_store import FaissVectorStore
from voice_ai_banking_support_agent.models import DocumentMetadata
from voice_ai_banking_support_agent.pipelines.build_dataset import _DedupJsonlAppender


def test_manifest_rejects_duplicate_bank_keys(tmp_path: Path) -> None:
    yaml_text = """
schema_version: "1"
banks:
  - bank_key: "acba"
    bank_name: "ACBA Bank"
    credits: { urls: ["https://a.test/credit"] }
    deposits: { urls: ["https://a.test/deposit"] }
    branches: { urls: ["https://a.test/branch"] }
  - bank_key: "acba"
    bank_name: "ACBA Duplicate"
    credits: { urls: ["https://b.test/credit"] }
    deposits: { urls: ["https://b.test/deposit"] }
    branches: { urls: ["https://b.test/branch"] }
"""
    p = tmp_path / "banks.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_banks_manifest(p)


def test_manifest_rejects_empty_topic_urls(tmp_path: Path) -> None:
    yaml_text = """
schema_version: "1"
banks:
  - bank_key: "acba"
    bank_name: "ACBA Bank"
    credits: { urls: [] }
    deposits: { urls: ["https://a.test/deposit"] }
    branches: { urls: ["https://a.test/branch"] }
"""
    p = tmp_path / "banks.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_banks_manifest(p)


def test_vector_store_bank_filter_accepts_bank_key(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    docs = [
        DocumentMetadata(
            bank_key="acba",
            bank_name="ACBA Bank",
            topic="credit",
            source_url="https://example.test/credit",
            page_title="Credit",
            section_title="Terms",
            language="hy",
            chunk_id="chunk-1",
            raw_text="raw",
            cleaned_text="credit terms",
        )
    ]
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    FaissVectorStore.build_and_save(
        embeddings=emb,
        docs=docs,
        index_dir=index_dir,
        index_name="t",
    )
    store = FaissVectorStore(
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.jsonl",
    )
    result = store.search(query_embedding=emb, top_k=1, bank_filter="acba")
    assert len(result) == 1


def test_dedup_jsonl_appender_avoids_duplicate_rows(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    writer = _DedupJsonlAppender()
    row = {"chunk_id": "c1", "text": "hello"}
    assert writer.append(p, [row], unique_key_fields=["chunk_id"]) == 1
    assert writer.append(p, [row], unique_key_fields=["chunk_id"]) == 0
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
