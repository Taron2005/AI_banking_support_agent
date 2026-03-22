from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from .config import AppConfig
from .indexing.embedder import EmbedderConfig, EmbeddingModel, resolve_embedding_device
from .indexing.vector_store import FaissVectorStore
from .models import DocumentMetadata, TopicLabel
from .pipelines.build_dataset import build_dataset
from .pipelines.build_index import build_index
from .pipelines.discover_urls import discover_urls
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _safe_console_text(text: str) -> str:
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline ingestion + indexing pipeline CLI")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    p.add_argument("--project-root", type=str, default=".", help="Repository root path.")
    p.add_argument("--log-level", type=str, default="INFO")

    sub = p.add_subparsers(dest="command", required=True)

    s_scrape = sub.add_parser("scrape", help="Scrape/ingest into local JSONL artifacts")
    s_scrape.add_argument("--banks", nargs="*", default=None, help="Bank keys, e.g. acba idbank")
    s_scrape.add_argument("--topics", nargs="+", default=["credit", "deposit", "branch"])

    s_build = sub.add_parser("build-index", help="Build local FAISS index from chunk JSONL")
    s_build.add_argument("--index-name", type=str, required=True)
    s_build.add_argument("--banks", nargs="*", default=None, help="Bank keys to include")
    s_build.add_argument("--topics", nargs="+", default=["credit", "deposit", "branch"])

    s_demo = sub.add_parser("demo-retrieve", help="Run retrieval demo against a built index")
    s_demo.add_argument("--index-name", type=str, required=True)
    s_demo.add_argument("--query", type=str, required=True)
    s_demo.add_argument("--top-k", type=int, default=5)
    s_demo.add_argument("--topic", type=str, default=None, help="Optional topic filter: credit/deposit/branch")
    s_demo.add_argument("--bank", type=str, default=None, help="Optional bank filter: bank_key or bank_name")

    s_inspect = sub.add_parser("inspect-doc", help="Print one stored chunk document by chunk_id")
    s_inspect.add_argument("--index-name", type=str, required=True)
    s_inspect.add_argument("--chunk-id", type=str, required=True)

    s_discover = sub.add_parser("discover-urls", help="Controlled same-domain URL discovery review mode")
    s_discover.add_argument("--banks", nargs="*", default=None, help="Optional bank keys to scope discovery")
    s_discover.add_argument("--max-pages", type=int, default=120)
    s_discover.add_argument("--max-depth", type=int, default=2)

    s_runtime_chat = sub.add_parser("runtime-chat", help="Run text runtime chat loop")
    s_runtime_chat.add_argument("--index-name", type=str, required=True)
    s_runtime_chat.add_argument("--runtime-config", type=str, default=None)
    s_runtime_chat.add_argument("--llm-config", type=str, default="llm_config.yaml")
    s_runtime_chat.add_argument("--verbose", action="store_true")

    s_runtime_eval = sub.add_parser("runtime-eval", help="Run curated runtime smoke evaluation")
    s_runtime_eval.add_argument("--index-name", type=str, required=True)
    s_runtime_eval.add_argument("--runtime-config", type=str, default=None)
    s_runtime_eval.add_argument("--llm-config", type=str, default="llm_config.yaml")
    s_runtime_eval.add_argument("--verbose", action="store_true")

    s_voice_agent = sub.add_parser("voice-agent", help="Run LiveKit voice agent (self-hosted)")
    s_voice_agent.add_argument("--index-name", type=str, required=True)
    s_voice_agent.add_argument("--runtime-config", type=str, default=None)
    s_voice_agent.add_argument("--llm-config", type=str, default="llm_config.yaml")
    s_voice_agent.add_argument("--voice-config", type=str, default=None)

    s_voice_smoke = sub.add_parser("voice-smoke-test", help="Run voice pipeline smoke flow with mock STT/TTS")
    s_voice_smoke.add_argument("--index-name", type=str, required=True)
    s_voice_smoke.add_argument("--runtime-config", type=str, default=None)
    s_voice_smoke.add_argument("--llm-config", type=str, default="llm_config.yaml")
    s_voice_smoke.add_argument("--voice-config", type=str, default=None)

    return p.parse_args(argv)


def _load_config(args: argparse.Namespace) -> AppConfig:
    from .config import load_config

    project_root = Path(args.project_root).resolve()
    config_yaml = Path(args.config).resolve() if args.config else None
    return load_config(project_root=project_root, config_yaml=config_yaml)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    setup_logging(args.log_level)
    cfg = _load_config(args)

    topics: list[TopicLabel] = [t.lower() for t in getattr(args, "topics", [])]  # type: ignore[list-item]

    if args.command == "scrape":
        build_dataset(config=cfg, banks=args.banks, topics=topics)
        return

    if args.command == "build-index":
        build_index(config=cfg, index_name=args.index_name, banks=args.banks, topics=topics)
        return

    if args.command == "demo-retrieve":
        index_dir = cfg.index_dir / args.index_name
        store = FaissVectorStore(
            index_path=index_dir / "faiss.index",
            metadata_path=index_dir / "metadata.jsonl",
        )

        topic_filter: TopicLabel | None = None
        if args.topic:
            topic_filter = args.topic.lower()  # validated by DocumentMetadata during build; retriever filter uses it

        emb_dev = resolve_embedding_device(cfg.embedding_device)
        embedder = EmbeddingModel(
            EmbedderConfig(
                model_name=cfg.embedding_model_name,
                device=emb_dev,
                batch_size=max(1, int(cfg.embedding_batch_size)),
                normalize=True,
            )
        )
        query_embedding: np.ndarray = embedder.embed_query(args.query)

        bank_keys = frozenset({args.bank.strip().lower()}) if args.bank else None
        results = store.search(
            query_embedding=query_embedding,
            top_k=args.top_k,
            topic_filter=topic_filter,
            bank_keys=bank_keys,
        )

        for i, r in enumerate(results, start=1):
            doc = r.doc
            print(f"#{i} score={r.score:.4f}")
            print(
                f"  bank={_safe_console_text(doc.bank_name)} ({doc.bank_key}) "
                f"topic={doc.topic} chunk_id={doc.chunk_id}"
            )
            print(f"  source_url={doc.source_url}")
            print(f"  page_title={_safe_console_text(doc.page_title)}")
            print(f"  section_title={_safe_console_text(doc.section_title)}")
            # Truncate long chunk output for CLI readability.
            cleaned = doc.cleaned_text
            if len(cleaned) > 600:
                cleaned = cleaned[:600] + "..."
            print(f"  cleaned_text={_safe_console_text(cleaned)}")
            print("")

        if not results:
            print("No retrieval results found with the applied filters.")
        return

    if args.command == "inspect-doc":
        index_dir = cfg.index_dir / args.index_name
        metadata_path = index_dir / "metadata.jsonl"
        target_id = args.chunk_id
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("chunk_id") == target_id:
                    doc = DocumentMetadata.model_validate(obj)
                    print(doc.model_dump_json(ensure_ascii=False, indent=2))
                    return
        print(f"chunk_id not found: {target_id}")
        return

    if args.command == "discover-urls":
        discover_urls(
            config=cfg,
            bank_keys=args.banks,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
        )
        return

    if args.command == "runtime-chat":
        from .runtime.cli_chat import run_chat

        run_chat(
            project_root=str(cfg.project_root),
            config_path=args.config,
            runtime_config_path=args.runtime_config,
            llm_config_path=args.llm_config,
            index_name=args.index_name,
            verbose=args.verbose,
        )
        return

    if args.command == "runtime-eval":
        from .runtime.eval_runtime import EVAL_QUERIES
        from .runtime.factory import build_runtime_orchestrator
        from .runtime.llm_config import load_llm_settings
        from .runtime.orchestrator import RuntimeRequest
        from .runtime.runtime_config import load_runtime_settings
        from .runtime.session_state import SessionStateStore

        runtime_settings = load_runtime_settings(args.runtime_config)
        llm_settings = load_llm_settings(args.llm_config)
        orchestrator = build_runtime_orchestrator(
            app_config=cfg, runtime_settings=runtime_settings, llm_settings=llm_settings
        )
        store = SessionStateStore()
        state = store.get_or_create("runtime-eval")
        for query in EVAL_QUERIES:
            out = orchestrator.handle(
                RuntimeRequest(
                    session_id="runtime-eval",
                    query=query,
                    index_name=args.index_name,
                    verbose=args.verbose,
                ),
                state,
            )
            print(_safe_console_text(json.dumps({"query": query, **out.model_dump()}, ensure_ascii=False)))
        return

    if args.command == "voice-agent":
        from .voice.cli import run_livekit_agent

        run_livekit_agent(
            project_root=str(cfg.project_root),
            app_config_path=args.config,
            runtime_config_path=args.runtime_config,
            llm_config_path=args.llm_config,
            voice_config_path=args.voice_config,
            index_name=args.index_name,
        )
        return

    if args.command == "voice-smoke-test":
        from .voice.cli import run_voice_smoke

        run_voice_smoke(
            project_root=str(cfg.project_root),
            app_config_path=args.config,
            runtime_config_path=args.runtime_config,
            llm_config_path=args.llm_config,
            voice_config_path=args.voice_config,
            index_name=args.index_name,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

