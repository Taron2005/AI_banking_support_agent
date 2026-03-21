from __future__ import annotations

import logging
from dataclasses import dataclass

from ..models import TopicLabel
from .answer_generator import AnswerBackend, GroundedAnswerGenerator
from .bank_detector import BankDetector
from .evidence_checker import EvidenceChecker
from .followup_resolver import FollowUpResolver
from .models import RuntimeResponse, SessionState
from .query_normalizer import normalize_query
from .refusal import refusal_message
from .retriever import RetrievalRequest, RuntimeRetriever
from .topic_classifier import TopicClassifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeRequest:
    session_id: str
    query: str
    index_name: str
    top_k: int = 8
    verbose: bool = False


class RuntimeOrchestrator:
    """Main text-runtime orchestration pipeline."""

    def __init__(
        self,
        *,
        retriever: RuntimeRetriever,
        topic_classifier: TopicClassifier | None = None,
        bank_detector: BankDetector | None = None,
        followup_resolver: FollowUpResolver | None = None,
        evidence_checker: EvidenceChecker | None = None,
        answer_generator: AnswerBackend | None = None,
        default_top_k: int = 8,
    ) -> None:
        self._retriever = retriever
        self._topic_classifier = topic_classifier or TopicClassifier()
        self._bank_detector = bank_detector or BankDetector()
        self._followup_resolver = followup_resolver or FollowUpResolver()
        self._evidence_checker = evidence_checker or EvidenceChecker()
        self._answer_generator = answer_generator or GroundedAnswerGenerator()
        self._default_top_k = default_top_k

    def handle(self, req: RuntimeRequest, state: SessionState) -> RuntimeResponse:
        trace: list[str] = []
        normalized = normalize_query(req.query)
        trace.append(f"normalized={normalized}")
        followup = self._followup_resolver.resolve(normalized, state)
        effective_query = followup.resolved_query
        trace.append(f"followup.used={followup.used_followup_context} merged={','.join(followup.merged_fields)}")
        topic = self._topic_classifier.classify(effective_query)
        trace.append(f"topic={topic.label} conf={topic.confidence:.2f} reason={topic.reason}")

        if topic.label in ("out_of_scope", "unsupported_request_type", "ambiguous"):
            reason = (
                "unsupported_request_type"
                if topic.label == "unsupported_request_type"
                else ("ambiguous" if topic.label == "ambiguous" else "out_of_scope")
            )
            self._update_state(state, normalized, "", None, None)
            return RuntimeResponse(
                answer_text=refusal_message(reason),
                status="refused",
                refusal_reason=reason,
                detected_topic=topic.label,
                decision_trace=trace if req.verbose else [],
            )

        detected_bank = self._bank_detector.detect(effective_query)
        bank_filter = detected_bank.bank_key if detected_bank else (state.last_bank if followup.used_followup_context else None)
        trace.append(f"bank_filter={bank_filter or 'none'}")
        runtime_topic = topic.label
        retrieved = self._retriever.retrieve(
            RetrievalRequest(
                query=effective_query,
                index_name=req.index_name,
                top_k=req.top_k or self._default_top_k,
                topic=runtime_topic,  # type: ignore[arg-type]
                bank_filter=bank_filter,
            )
        )
        trace.append(f"retrieved_count={len(retrieved)}")

        evidence = self._evidence_checker.assess(effective_query, runtime_topic, retrieved)
        trace.append(f"evidence.sufficient={evidence.sufficient} reason={evidence.reason} max_score={evidence.max_score:.3f}")
        if not evidence.sufficient:
            self._update_state(state, normalized, "", bank_filter, runtime_topic)  # keep context
            return RuntimeResponse(
                answer_text=refusal_message("insufficient_evidence"),
                status="refused",
                refusal_reason="insufficient_evidence",
                detected_topic=runtime_topic,
                detected_bank=bank_filter,
                retrieved_chunks_summary=[f"{c.chunk.chunk_id}:{c.score:.3f}" for c in retrieved[:3]],
                decision_trace=trace if req.verbose else [],
            )

        answer = self._answer_generator.generate(effective_query, runtime_topic, retrieved, bank_filter)
        self._update_state(state, normalized, answer, bank_filter, runtime_topic)  # type: ignore[arg-type]
        return RuntimeResponse(
            answer_text=answer,
            status="answered",
            detected_topic=runtime_topic,
            detected_bank=bank_filter,
            used_sources=[c.chunk.source_url for c in retrieved[:3]],
            retrieved_chunks_summary=[
                f"{c.chunk.bank_key}/{c.chunk.topic} {c.score:.3f} {c.chunk.section_title}" for c in retrieved[:3]
            ],
            state_updates={
                "last_topic": runtime_topic,
                "last_bank": bank_filter or "",
            },
            decision_trace=trace if req.verbose else [],
        )

    @staticmethod
    def _update_state(
        state: SessionState,
        user_turn: str,
        assistant_turn: str,
        bank: str | None,
        topic: TopicLabel | None,
    ) -> None:
        state.recent_user_turns = (state.recent_user_turns + [user_turn])[-5:]
        if assistant_turn:
            state.recent_assistant_turns = (state.recent_assistant_turns + [assistant_turn])[-5:]
        if bank:
            state.last_bank = bank
        if topic:
            state.last_topic = topic
        low = user_turn.lower()
        for city in ("երևան", "գյումրի", "gyumri", "yerevan"):
            if city in low:
                state.last_city = city
                break
        for product in ("սպառողական վարկ", "consumer loan", "ավանդ", "deposit", "մասնաճյուղ", "branch"):
            if product in low:
                state.last_product = product
                break

