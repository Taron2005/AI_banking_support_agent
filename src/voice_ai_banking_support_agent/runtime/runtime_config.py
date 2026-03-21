from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class TopicClassifierSettings(BaseModel):
    ambiguous_margin: float = 0.2
    strong_term_weight: float = 1.5
    weak_term_weight: float = 0.75


class FollowUpSettings(BaseModel):
    short_query_max_tokens: int = 6
    markers: list[str] = Field(
        default_factory=lambda: ["իսկ", "and", "what about", "and for", "then", "also"]
    )
    city_terms: list[str] = Field(default_factory=lambda: ["երևան", "gyumri", "գյումրի", "vanadzor"])


class EvidenceSettings(BaseModel):
    min_chunks: int = 1
    min_top_score: float = 0.15
    branch_address_patterns: list[str] = Field(default_factory=lambda: ["հասցե", "address", "street", "փողոց"])


class AnswerSettings(BaseModel):
    backend: Literal["extractive", "llm"] = "llm"
    max_evidence_chunks: int = 4
    max_snippet_chars: int = 520
    llm_model_name: str | None = None


class RetrievalSettings(BaseModel):
    default_top_k: int = 8


class RuntimeSettings(BaseModel):
    topic_classifier: TopicClassifierSettings = Field(default_factory=TopicClassifierSettings)
    followup: FollowUpSettings = Field(default_factory=FollowUpSettings)
    evidence: EvidenceSettings = Field(default_factory=EvidenceSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    bank_aliases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "ameriabank": ["ameriabank", "ameria", "ամերիա", "ամերիաբանկ"],
            "acba": ["acba", "ակբա", "acba bank"],
            "idbank": ["idbank", "id bank", "այդիբանկ", "իդբանկ"],
        }
    )


def load_runtime_settings(path: str | Path | None = None) -> RuntimeSettings:
    if path is None:
        return RuntimeSettings()
    p = Path(path)
    if not p.exists():
        return RuntimeSettings()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return RuntimeSettings.model_validate(data)

