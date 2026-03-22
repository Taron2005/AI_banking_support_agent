from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class TopicClassifierSettings(BaseModel):
    ambiguous_margin: float = 0.2
    strong_term_weight: float = 1.5
    weak_term_weight: float = 0.75
    # Match STT/typo variants (SequenceMatcher on tokens).
    fuzzy_match: bool = True
    fuzzy_ratio: float = 0.8


class FollowUpSettings(BaseModel):
    short_query_max_tokens: int = 6
    markers: list[str] = Field(
        default_factory=lambda: ["իսկ", "and", "what about", "and for", "then", "also"]
    )
    city_terms: list[str] = Field(default_factory=lambda: ["երևան", "gyumri", "գյումրի", "vanadzor"])


class EvidenceSettings(BaseModel):
    min_chunks: int = 1
    min_top_score: float = 0.15
    branch_address_patterns: list[str] = Field(
        default_factory=lambda: ["հասցե", "address", "street", "փողոց", "ք.", "շենք", "պող.", "հՀ,"]
    )


class AnswerSettings(BaseModel):
    backend: Literal["extractive", "llm"] = "llm"
    max_evidence_chunks: int = 5
    max_snippet_chars: int = 600
    # Per-chunk text sent to the LLM (larger = more grounding detail for synthesis).
    max_chars_per_evidence: int = 900
    llm_model_name: str | None = None


class RetrievalSettings(BaseModel):
    default_top_k: int = 8
    # Wider FAISS candidate pool before lexical rerank / hybrid fusion (dense is cheap vs LLM).
    faiss_candidate_pool_factor: int = Field(default=8, ge=4, le=24)
    # Dense (FAISS cosine) + sparse (BM25) fusion on the FAISS candidate pool.
    hybrid_bm25: bool = True
    hybrid_dense_weight: float = Field(default=0.58, ge=0.0, le=1.0)
    # When no single-bank scope: cap chunks per bank / per canonical URL in the rerank pass so one bank or one hub page cannot dominate.
    diversify_max_per_bank: int = Field(default=2, ge=1, le=8)
    # 0 = disable per-URL cap (only bank cap applies).
    diversify_max_per_source_url: int = Field(default=2, ge=0, le=8)
    # Optional cross-encoder rerank (multilingual); adds latency on first load.
    cross_encoder_model: str | None = "cross-encoder/ms-marco-MultilingualBERT-L-12"
    cross_encoder_top_n: int = Field(default=24, ge=4, le=96)
    cross_encoder_enabled: bool = False


class OrchestrationSettings(BaseModel):
    """
    Strict gates for evaluation-ready behavior (defaults preserve legacy multi-bank retrieval).
    """

    # If true: credit/deposit/branch queries without a resolved bank and without
    # "all banks" / comparison wording → clarify (no retrieval).
    require_explicit_bank: bool = False
    # If true: when user did not ask for all banks or comparison, collapse retrieved chunks
    # to the single highest-scoring bank before LLM (reduces cross-bank mixing).
    restrict_evidence_to_single_bank_without_comparison: bool = False
    # If true: after retrieval, if the query is not all-banks / comparison-scoped but hits
    # multiple banks, ask which bank instead of answering from mixed evidence.
    clarify_when_unscoped_multi_bank_evidence: bool = False
    # If true: explicit comparison queries need evidence from at least two banks or we refuse.
    refuse_comparison_without_multi_bank_evidence: bool = False


class RuntimeSettings(BaseModel):
    topic_classifier: TopicClassifierSettings = Field(default_factory=TopicClassifierSettings)
    followup: FollowUpSettings = Field(default_factory=FollowUpSettings)
    evidence: EvidenceSettings = Field(default_factory=EvidenceSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    orchestration: OrchestrationSettings = Field(default_factory=OrchestrationSettings)
    bank_aliases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "ameriabank": [
                "ameriabank",
                "ameria bank",
                "ameria",
                "ամերիա",
                "ամերիաբանկ",
                "ամերիա բանկ",
                "ամերիայի",
                "ամերիաբանկի",
            ],
            "acba": ["acba", "acba bank", "ակբա", "ակբա բանկ"],
            "idbank": ["idbank", "id bank", "այդիբանկ", "իդբանկ", "այդի բանկ"],
        }
    )


def load_runtime_settings(path: str | Path | None = None) -> RuntimeSettings:
    if path is None:
        cfg = RuntimeSettings()
    else:
        p = Path(path)
        if not p.exists():
            cfg = RuntimeSettings()
        else:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cfg = RuntimeSettings.model_validate(data)

    hb = (os.getenv("RETRIEVAL_HYBRID_BM25") or "").strip().lower()
    if hb in ("0", "false", "no", "off"):
        cfg = cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update={"hybrid_bm25": False})})
    elif hb in ("1", "true", "yes", "on"):
        cfg = cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update={"hybrid_bm25": True})})

    ce = (os.getenv("RETRIEVAL_CROSS_ENCODER") or "").strip().lower()
    if ce in ("1", "true", "yes", "on"):
        cfg = cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update={"cross_encoder_enabled": True})})
    elif ce in ("0", "false", "no", "off"):
        cfg = cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update={"cross_encoder_enabled": False})})

    cem = (os.getenv("CROSS_ENCODER_MODEL") or "").strip()
    if cem:
        cfg = cfg.model_copy(
            update={"retrieval": cfg.retrieval.model_copy(update={"cross_encoder_model": cem})}
        )

    return cfg

