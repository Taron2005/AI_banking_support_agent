from __future__ import annotations

from ..config import AppConfig
from .answer_generator import AnswerGeneratorConfig, GroundedAnswerGenerator, LLMAnswerGenerator
from .bank_detector import BankDetector
from .evidence_checker import EvidenceChecker, EvidenceConfig
from .followup_resolver import FollowUpResolver
from .llm import build_llm_client
from .llm_config import LLMSettings
from .orchestrator import RuntimeOrchestrator
from .retriever import RuntimeRetriever
from .runtime_config import RuntimeSettings
from .topic_classifier import TopicClassifier, TopicClassifierConfig


def build_runtime_orchestrator(
    *,
    app_config: AppConfig,
    runtime_settings: RuntimeSettings,
    llm_client: object | None = None,
    llm_settings: LLMSettings | None = None,
) -> RuntimeOrchestrator:
    retriever = RuntimeRetriever(app_config, runtime_settings.retrieval)
    topic_classifier = TopicClassifier(
        TopicClassifierConfig(
            ambiguous_margin=runtime_settings.topic_classifier.ambiguous_margin,
            strong_term_weight=runtime_settings.topic_classifier.strong_term_weight,
            weak_term_weight=runtime_settings.topic_classifier.weak_term_weight,
            fuzzy_match=runtime_settings.topic_classifier.fuzzy_match,
            fuzzy_ratio=runtime_settings.topic_classifier.fuzzy_ratio,
        )
    )
    bank_detector = BankDetector(aliases=runtime_settings.bank_aliases)
    followup = FollowUpResolver(
        followup_markers=runtime_settings.followup.markers,
        short_query_max_tokens=runtime_settings.followup.short_query_max_tokens,
        city_terms=runtime_settings.followup.city_terms,
        bank_aliases=runtime_settings.bank_aliases,
    )
    evidence = EvidenceChecker(
        EvidenceConfig(
            min_chunks=runtime_settings.evidence.min_chunks,
            min_top_score=runtime_settings.evidence.min_top_score,
            branch_address_patterns=tuple(runtime_settings.evidence.branch_address_patterns),
        )
    )
    answer_cfg = AnswerGeneratorConfig(
        max_evidence_chunks=runtime_settings.answer.max_evidence_chunks,
        max_snippet_chars=runtime_settings.answer.max_snippet_chars,
        max_chars_per_evidence=runtime_settings.answer.max_chars_per_evidence,
    )
    extractive = GroundedAnswerGenerator(answer_cfg)
    if runtime_settings.answer.backend == "llm":
        effective_llm_client = llm_client or build_llm_client(llm_settings)
        answer_backend = LLMAnswerGenerator(
            llm_client=effective_llm_client, fallback=extractive, cfg=answer_cfg
        )
    else:
        answer_backend = extractive
    return RuntimeOrchestrator(
        retriever=retriever,
        topic_classifier=topic_classifier,
        bank_detector=bank_detector,
        followup_resolver=followup,
        evidence_checker=evidence,
        answer_generator=answer_backend,
        default_top_k=runtime_settings.retrieval.default_top_k,
        max_evidence_chunks=runtime_settings.answer.max_evidence_chunks,
        orchestration=runtime_settings.orchestration,
        bank_aliases=runtime_settings.bank_aliases,
    )

