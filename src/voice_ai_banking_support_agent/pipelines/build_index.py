from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..indexing.embedder import EmbedderConfig, EmbeddingModel
from ..indexing.vector_store import FaissVectorStore
from ..models import DocumentMetadata, TopicLabel
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _safe_for_log(text: str) -> str:
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def _read_chunk_jsonl(path: Path) -> list[DocumentMetadata]:
    if not path.exists():
        return []
    docs: list[DocumentMetadata] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                docs.append(DocumentMetadata.model_validate(obj))
            except Exception as exc:
                logger.warning(
                    "Skipping malformed chunk row file=%s error=%s",
                    path,
                    _safe_for_log(str(exc)),
                )
    return docs


def build_index(
    *,
    config: AppConfig,
    index_name: str,
    banks: list[str] | None,
    topics: list[TopicLabel],
) -> None:
    """Build a persistent local FAISS index from chunk artifacts."""

    setup_logging()
    config.ensure_dirs()

    selected_banks = {b.lower() for b in banks} if banks else None
    index_dir = config.index_dir / index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    all_docs: list[DocumentMetadata] = []
    chunks_per_file: dict[str, int] = {}

    # Read chunk artifacts for the selected banks/topics.
    chunk_dir = config.chunks_dir
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunks directory does not exist: {chunk_dir}")

    chunk_files = sorted(chunk_dir.glob("*_chunks.jsonl"))
    for file in chunk_files:
        # Filename: {bank_key}_{topic}_chunks.jsonl
        name = file.name
        if not name.endswith("_chunks.jsonl"):
            continue
        parts = name.removesuffix("_chunks.jsonl").split("_")
        if len(parts) < 2:
            continue
        bank_key = "_".join(parts[:-1])
        topic = parts[-1]

        if selected_banks is not None and bank_key.lower() not in selected_banks:
            continue
        if topic not in topics:
            continue

        rows = _read_chunk_jsonl(file)
        rel = str(file.relative_to(chunk_dir))
        chunks_per_file[rel] = len(rows)
        all_docs.extend(rows)

    if not all_docs:
        raise ValueError("No chunk documents found for the selected banks/topics.")

    deduped: dict[str, DocumentMetadata] = {}
    for doc in all_docs:
        deduped[doc.chunk_id] = doc
    if len(deduped) != len(all_docs):
        logger.info("Dropped duplicate chunks before embedding: %d -> %d", len(all_docs), len(deduped))
    all_docs = list(deduped.values())

    # Stable row order: FAISS ids must align with metadata.jsonl across machines and rebuilds.
    all_docs.sort(
        key=lambda d: (
            (d.bank_key or "").lower(),
            (d.topic or "").lower(),
            (d.source_url or "").lower(),
            (d.chunk_id or "").lower(),
        )
    )

    by_bank_topic = Counter((d.bank_key, d.topic) for d in all_docs)
    banks_present = sorted({d.bank_key for d in all_docs if d.bank_key})
    topics_present = sorted({d.topic for d in all_docs if d.topic})
    logger.info(
        "Chunk corpus: total=%d banks=%s topics=%s by_bank_topic=%s",
        len(all_docs),
        banks_present,
        topics_present,
        {f"{b}/{t}": n for (b, t), n in sorted(by_bank_topic.items())},
    )
    for rel, n in sorted(chunks_per_file.items()):
        logger.info("  %s: %d rows", rel, n)

    if selected_banks:
        found_banks = {bk.lower() for bk in banks_present}
        for want in selected_banks:
            if want not in found_banks:
                logger.warning(
                    "No chunks found for bank_key=%r in this build (check scrape/chunks for that bank).",
                    want,
                )

    logger.info("Embedding %d chunk documents...", len(all_docs))
    embedder = EmbeddingModel(
        EmbedderConfig(model_name=config.embedding_model_name, device="cpu", batch_size=32)
    )
    try:
        embeddings = embedder.embed_texts([d.cleaned_text for d in all_docs])
    except Exception as e:
        # Embedding errors can come from transitive ML stack dependencies (TF/protobuf, etc).
        # Provide a short actionable message.
        raise RuntimeError(
            "Failed to compute embeddings. If you see TensorFlow/protobuf errors, "
            "ensure TF is disabled (TRANSFORMERS_NO_TF=1) or protobuf is compatible. "
            "This project already attempts to disable TF, but your environment may still "
            "require a pip/env adjustment. Original error: "
            + str(e)
        ) from e

    emb_dim = int(embeddings.shape[1])
    if emb_dim <= 0:
        raise ValueError("Embedding dimension must be positive.")

    extra_index_info: dict[str, object] = {
        "banks_present": banks_present,
        "topics_present": topics_present,
        "chunks_by_bank_topic": {f"{b}/{t}": n for (b, t), n in sorted(by_bank_topic.items())},
        "chunk_source_files": chunks_per_file,
    }

    # Persist FAISS + metadata mapping.
    FaissVectorStore.build_and_save(
        embeddings=embeddings,
        docs=all_docs,
        index_dir=index_dir,
        index_name=index_name,
        embedding_model_name=config.embedding_model_name,
        extra_index_info=extra_index_info,
    )

    logger.info(
        "Index written to %s (vectors=%d dim=%d model=%r)",
        index_dir,
        len(all_docs),
        emb_dim,
        config.embedding_model_name,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build vector index from chunk artifacts.")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    p.add_argument("--project-root", type=str, default=".", help="Repository root path.")
    p.add_argument("--index-name", type=str, required=True)
    p.add_argument("--banks", nargs="*", default=None, help="Bank keys to include, e.g. acba.")
    p.add_argument("--topics", nargs="+", default=["credit", "deposit", "branch"])
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    setup_logging(args.log_level)

    project_root = Path(args.project_root).resolve()
    config_yaml = Path(args.config).resolve() if args.config else None

    from ..config import load_config

    cfg = load_config(project_root=project_root, config_yaml=config_yaml)
    topics: list[TopicLabel] = [t.lower() for t in args.topics]  # type: ignore[list-item]
    build_index(config=cfg, index_name=args.index_name, banks=args.banks, topics=topics)


if __name__ == "__main__":
    main()

