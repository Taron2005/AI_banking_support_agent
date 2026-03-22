from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_EMBEDDING_MODEL = "Metric-AI/armenian-text-embeddings-2-large"


@dataclass(frozen=True)
class NetworkConfig:
    """Network settings for offline scraping."""

    timeout_seconds: float = 30.0
    retries: int = 3
    backoff_min_seconds: float = 1.0
    backoff_max_seconds: float = 10.0
    # Polite pause before each live HTTP request (skipped when reusing cached raw HTML). Set via YAML/env for production scrapes.
    request_delay_seconds: float = 0.0
    user_agent: str = (
        "voice-ai-banking-support-agent/0.1 (+https://example.invalid; contact: offline indexing)"
    )


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking settings for RAG readiness."""

    # Soft target chunk size; the chunker will try to keep chunks near this size.
    target_words: int = 250
    min_words: int = 80
    max_words: int = 450
    # Number of trailing sentences carried into the next chunk for continuity (0 = none).
    overlap_sentences: int = 2

    # Hard cap on the number of sections extracted from a single page (safety guard).
    max_sections_per_page: int = 50


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration for scraping + indexing."""

    project_root: Path
    manifest_path: Path

    data_dir: Path
    raw_html_dir: Path
    branches_dir: Path
    cleaned_docs_dir: Path
    chunks_dir: Path
    index_dir: Path

    language: Literal["hy"] = "hy"
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    # auto | cpu | cuda | mps — passed to SentenceTransformer for build-index and runtime query embedding.
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    # Optional GPU-backed FAISS for search (requires faiss-gpu / GPU build of FAISS). Index files stay CPU-compatible.
    faiss_use_gpu: bool = False
    faiss_gpu_id: int = 0

    chunking: ChunkingConfig = ChunkingConfig()
    network: NetworkConfig = NetworkConfig()

    # If true, the dataset build will not overwrite existing raw HTML artifacts.
    reuse_raw_html: bool = True

    def ensure_dirs(self) -> None:
        """Create all configured output directories if they do not exist."""

        for p in [
            self.data_dir,
            self.raw_html_dir,
            self.branches_dir,
            self.cleaned_docs_dir,
            self.chunks_dir,
            self.index_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


def _deep_update_dataclass(obj: Any, updates: dict[str, Any]) -> Any:
    """
    Very small helper to update nested dataclass instances.

    Notes:
    - This is intentionally minimal (avoid overengineering config frameworks).
    - Only supports dict keys matching dataclass field names; unknown keys are ignored.
    """

    if not updates:
        return obj

    # If nested dataclass exists in updates, update it recursively.
    if not hasattr(obj, "__dataclass_fields__"):
        return obj

    kwargs: dict[str, Any] = {}
    for field_name, field_def in obj.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if field_name in updates:
            value = updates[field_name]
            field_type = field_def.type
            if hasattr(getattr(obj, field_name), "__dataclass_fields__") and isinstance(value, dict):
                kwargs[field_name] = _deep_update_dataclass(getattr(obj, field_name), value)
            else:
                kwargs[field_name] = value
        else:
            kwargs[field_name] = getattr(obj, field_name)
    return type(obj)(**kwargs)


def load_config(project_root: Path, config_yaml: Path | None = None) -> AppConfig:
    """
    Load an `AppConfig`.

    Args:
        project_root: Root directory of the repository.
        config_yaml: Optional YAML path that can override select fields.

    Returns:
        A configured `AppConfig`.
    """

    manifest_path = project_root / "manifests" / "banks.yaml"
    data_dir = project_root / "data"

    cfg = AppConfig(
        project_root=project_root,
        manifest_path=manifest_path,
        data_dir=data_dir,
        raw_html_dir=data_dir / "raw_html",
        branches_dir=data_dir / "branches",
        cleaned_docs_dir=data_dir / "cleaned_docs",
        chunks_dir=data_dir / "chunks",
        index_dir=data_dir / "index",
    )

    if config_yaml is None:
        return cfg

    if not config_yaml.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_yaml}")

    overrides = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(overrides, dict):
        raise ValueError("Config YAML root must be a mapping/object.")

    # Allow env var overrides for quick experimentation / CI.
    embedding_model = overrides.get("embedding_model_name") or os.getenv("EMBEDDING_MODEL_NAME")
    if embedding_model:
        overrides["embedding_model_name"] = embedding_model
    if os.getenv("EMBEDDING_DEVICE"):
        overrides["embedding_device"] = os.getenv("EMBEDDING_DEVICE", "auto").strip()
    if os.getenv("EMBEDDING_BATCH_SIZE"):
        overrides["embedding_batch_size"] = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    if os.getenv("FAISS_USE_GPU", "").strip().lower() in ("1", "true", "yes"):
        overrides["faiss_use_gpu"] = True
    if os.getenv("FAISS_GPU_ID"):
        overrides["faiss_gpu_id"] = int(os.getenv("FAISS_GPU_ID", "0"))
    if os.getenv("SCRAPER_TIMEOUT_SECONDS"):
        overrides.setdefault("network", {})
        overrides["network"]["timeout_seconds"] = float(os.getenv("SCRAPER_TIMEOUT_SECONDS", "30"))
    if os.getenv("SCRAPER_RETRIES"):
        overrides.setdefault("network", {})
        overrides["network"]["retries"] = int(os.getenv("SCRAPER_RETRIES", "3"))
    if os.getenv("SCRAPER_USER_AGENT"):
        overrides.setdefault("network", {})
        overrides["network"]["user_agent"] = os.getenv("SCRAPER_USER_AGENT")
    if os.getenv("SCRAPER_REQUEST_DELAY_SECONDS"):
        overrides.setdefault("network", {})
        overrides["network"]["request_delay_seconds"] = float(
            os.getenv("SCRAPER_REQUEST_DELAY_SECONDS", "0")
        )

    # Apply nested updates for network/chunking.
    network_updates = overrides.get("network")
    chunking_updates = overrides.get("chunking")
    if isinstance(network_updates, dict):
        overrides["network"] = _deep_update_dataclass(cfg.network, network_updates)
    if isinstance(chunking_updates, dict):
        overrides["chunking"] = _deep_update_dataclass(cfg.chunking, chunking_updates)

    # Build a new AppConfig by copying known fields.
    cfg_dict = cfg.__dict__.copy()
    for k, v in overrides.items():
        if k in cfg_dict:
            cfg_dict[k] = v

    # Normalize path-like fields to Path objects.
    path_fields = [
        "project_root",
        "manifest_path",
        "data_dir",
        "raw_html_dir",
        "branches_dir",
        "cleaned_docs_dir",
        "chunks_dir",
        "index_dir",
    ]
    for field in path_fields:
        value = cfg_dict.get(field)
        if isinstance(value, str):
            cfg_dict[field] = Path(value)

    # Resolve relative paths against project root for predictable behavior.
    for field in path_fields:
        value = cfg_dict.get(field)
        if isinstance(value, Path) and not value.is_absolute():
            cfg_dict[field] = (project_root / value).resolve()

    return AppConfig(**cfg_dict)  # type: ignore[arg-type]

