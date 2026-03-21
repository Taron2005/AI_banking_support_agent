from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..models import DocumentMetadata, TopicLabel

RuntimeStatus = Literal["answered", "refused", "clarify"]
RefusalReason = Literal[
    "out_of_scope",
    "unsupported_request_type",
    "insufficient_evidence",
    "comparison_insufficient",
    "prompt_injection",
    "ambiguous",
]
RuntimeTopic = Literal["credit", "deposit", "branch", "out_of_scope", "ambiguous", "unsupported_request_type"]


class TopicClassification(BaseModel):
    label: RuntimeTopic
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    matched_terms: list[str] = Field(default_factory=list)
    reason: str = ""


class FollowUpResolution(BaseModel):
    resolved_query: str
    used_followup_context: bool = False
    merged_fields: list[str] = Field(default_factory=list)
    needs_clarification: bool = False


class RetrievedChunk(BaseModel):
    score: float
    chunk: DocumentMetadata


class EvidenceDecision(BaseModel):
    sufficient: bool
    reason: str
    matched_chunk_count: int = 0
    min_score: float = 0.0
    max_score: float = 0.0


class RuntimeResponse(BaseModel):
    answer_text: str
    status: RuntimeStatus
    # How the answer was produced: llm | extractive_fallback | extractive_only (backend=extractive).
    answer_synthesis: str | None = None
    # When answer_synthesis=extractive_fallback, short reason (e.g. HTTPError, empty response).
    llm_error: str | None = None
    refusal_reason: RefusalReason | None = None
    detected_topic: RuntimeTopic | None = None
    # Single-bank shortcut for clients; None when zero or multiple banks apply.
    detected_bank: str | None = None
    # Populated whenever bank allowlist is non-empty (one or more keys).
    detected_banks: list[str] = Field(default_factory=list)
    used_sources: list[str] = Field(default_factory=list)
    retrieved_chunks_summary: list[str] = Field(default_factory=list)
    state_updates: dict[str, str] = Field(default_factory=dict)
    decision_trace: list[str] = Field(default_factory=list)


@dataclass
class SessionState:
    session_id: str
    last_topic: TopicLabel | None = None
    last_bank: str | None = None
    # Last explicit bank allowlist (e.g. comparison); used for "the other bank" style follow-ups.
    last_bank_keys: list[str] = field(default_factory=list)
    last_city: str | None = None
    last_product: str | None = None
    last_entities_mentioned: list[str] = field(default_factory=list)
    recent_user_turns: list[str] = field(default_factory=list)
    recent_assistant_turns: list[str] = field(default_factory=list)

