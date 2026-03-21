from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

TopicLabel = Literal["credit", "deposit", "branch"]


class SourcePage(BaseModel):
    """Raw/cleaned source page artifact produced by ingestion."""

    bank_key: str = ""
    bank_name: str
    topic: TopicLabel
    source_url: str
    page_title: str
    language: str
    raw_text: str
    cleaned_text: str


class CleanDocument(BaseModel):
    """Alias model for a cleaned page-level record."""

    bank_key: str = ""
    bank_name: str
    topic: TopicLabel
    source_url: str
    page_title: str
    language: str
    raw_text: str
    cleaned_text: str


class DocumentMetadata(BaseModel):
    """
    Metadata for a retrieval chunk.

    The intention is to be query-result explainable: each chunk can be traced back
    to its bank/topic/source page and section.
    """

    bank_key: str = ""
    bank_name: str
    topic: TopicLabel
    source_url: str
    page_title: str
    section_title: str
    language: str

    chunk_id: str

    # Store both raw extracted page text and the cleaned chunk text.
    raw_text: str
    cleaned_text: str

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> TopicLabel:
        v_lower = v.strip().lower()
        allowed = {"credit", "deposit", "branch"}
        if v_lower not in allowed:
            raise ValueError(f"Invalid topic label: {v}. Must be one of {sorted(allowed)}.")
        return v_lower  # type: ignore[return-value]


class BranchRecord(BaseModel):
    """Structured branch/location data extracted from branch pages."""

    bank_key: str = ""
    bank_name: str
    branch_name: str
    city: str
    district: Optional[str] = None
    address: str
    working_hours: Optional[str] = None
    phone: Optional[str] = None
    source_url: str


class ManifestBankTopic(BaseModel):
    """Allowed URLs for a topic for one bank."""

    urls: list[str] = Field(default_factory=list)


class ManifestBank(BaseModel):
    """Single bank entry from the manifest."""

    bank_key: str
    bank_name: str
    language: str = "hy"
    credits: ManifestBankTopic = Field(default_factory=ManifestBankTopic)
    deposits: ManifestBankTopic = Field(default_factory=ManifestBankTopic)
    branches: ManifestBankTopic = Field(default_factory=ManifestBankTopic)


class BanksManifest(BaseModel):
    """Top-level manifest container."""

    schema_version: str = "1"
    banks: list[ManifestBank] = Field(default_factory=list)


class ChunkRecord(DocumentMetadata):
    """Explicit chunk record model for storage/readability."""


class RetrievalResultModel(BaseModel):
    """Serializable retrieval output shape for debugging/reporting."""

    score: float
    chunk: DocumentMetadata

