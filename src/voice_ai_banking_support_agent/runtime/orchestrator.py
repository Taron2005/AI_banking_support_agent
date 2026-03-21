from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..models import TopicLabel
from .answer_generator import (
    AnswerBackend,
    AnswerMode,
    AnswerResult,
    GroundedAnswerGenerator,
    LLMAnswerGenerator,
)
from .evidence_pack import prepare_evidence_for_answer
from .bank_detector import BankDetector, BankMatch
from .bank_scope import query_implies_all_banks, query_implies_comparison
from .evidence_checker import EvidenceChecker
from .evidence_select import dedupe_urls
from .followup_resolver import FollowUpResolver
from .models import RuntimeResponse, SessionState
from .query_normalizer import normalize_query
from .orchestration_policy import dominant_bank_key_from_chunks, supported_bank_keys_csv
from .rag_prompts import REFUSAL_RULES
from .refusal import bank_clarification_message, refusal_message
from .retriever import RetrievalRequest, RuntimeRetriever
from .runtime_config import OrchestrationSettings
from .topic_classifier import TopicClassifier

logger = logging.getLogger(__name__)

_OTHER_BANK_RX = re.compile(
    r"(?:\b(?:the\s+)?other\s+banks?\b|"
    r"\b(?:the\s+)?other\s+one\b|"
    r"\bcompared\s+to\s+the\s+other\b|"
    r"մյուս\s+բանկ|մյուսը|մյուս\s+մեկը|"
    r"մյուս\s+բանկում|նրանց\s+մասին)",
    re.IGNORECASE,
)


def _expand_implicit_bank_matches(
    text: str,
    matches: list[BankMatch],
    state: SessionState,
) -> list[BankMatch]:
    """
    If the user refers to “the other bank” without naming it, reuse the last multi-bank scope.
    """

    if not _OTHER_BANK_RX.search(text):
        return matches
    prev = [str(k).strip().lower() for k in state.last_bank_keys if str(k).strip()]
    if len(prev) < 2:
        return matches
    have = {m.bank_key for m in matches}
    extra = [BankMatch(bank_key=k, matched_alias=k) for k in prev if k not in have]
    if not extra:
        return matches
    return list(matches) + extra


def _resolve_bank_keys(
    *,
    effective_query: str,
    detected_all: list[BankMatch],
    followup_merged_fields: list[str],
    state: SessionState,
) -> frozenset[str] | None:
    lower = effective_query.lower()
    # Two+ explicit bank names win over broad “all banks / compare” wording.
    if len(detected_all) >= 2:
        return frozenset(m.bank_key for m in detected_all)
    if query_implies_all_banks(lower):
        return None
    if len(detected_all) == 1:
        return frozenset({detected_all[0].bank_key})
    if followup_merged_fields and "last_bank" in followup_merged_fields and state.last_bank:
        return frozenset({state.last_bank})
    return None


def _distinct_bank_keys_in_retrieval(retrieved: list) -> set[str]:
    s: set[str] = set()
    for c in retrieved:
        k = (c.chunk.bank_key or "").strip().lower()
        if k:
            s.add(k)
    return s


def _response_bank_fields(bank_keys: frozenset[str] | None) -> tuple[str | None, list[str]]:
    if not bank_keys:
        return None, []
    ordered = sorted(bank_keys)
    if len(ordered) == 1:
        return ordered[0], ordered
    return None, ordered


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
        max_evidence_chunks: int = 5,
        orchestration: OrchestrationSettings | None = None,
        bank_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self._retriever = retriever
        self._topic_classifier = topic_classifier or TopicClassifier()
        self._bank_detector = bank_detector or BankDetector()
        self._followup_resolver = followup_resolver or FollowUpResolver()
        self._evidence_checker = evidence_checker or EvidenceChecker()
        self._answer_generator = answer_generator or GroundedAnswerGenerator()
        self._default_top_k = default_top_k
        self._max_evidence_chunks = max(1, int(max_evidence_chunks))
        self._orchestration = orchestration or OrchestrationSettings()
        self._bank_aliases = bank_aliases or getattr(bank_detector, "_aliases", {})

    def handle(self, req: RuntimeRequest, state: SessionState) -> RuntimeResponse:
        trace: list[str] = []
        normalized = normalize_query(req.query)
        trace.append(f"normalized={normalized}")
        followup = self._followup_resolver.resolve(normalized, state)
        effective_query = followup.resolved_query
        trace.append(f"followup.used={followup.used_followup_context} merged={','.join(followup.merged_fields)}")
        if followup.needs_clarification:
            hint = (
                "Խնդրում եմ նախորդ հարցից հետո կրկնել հարցը ավելի կոնկրետ՝ նշելով բանկը և թեման "
                "(վարկ, ավանդ կամ մասնաճյուղ)։"
            )
            self._update_state(state, normalized, "", None, None)
            return RuntimeResponse(
                answer_text=hint,
                status="clarify",
                refusal_reason=None,
                detected_topic=None,
                decision_trace=trace if req.verbose else [],
            )
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

        detected_all = _expand_implicit_bank_matches(
            effective_query,
            self._bank_detector.detect_all(effective_query),
            state,
        )
        bank_keys = _resolve_bank_keys(
            effective_query=effective_query,
            detected_all=detected_all,
            followup_merged_fields=followup.merged_fields,
            state=state,
        )
        trace.append(f"bank_keys={','.join(sorted(bank_keys)) if bank_keys else 'none'}")
        low_q = effective_query.lower()
        if self._orchestration.require_explicit_bank:
            if (
                bank_keys is None
                and not query_implies_all_banks(low_q)
                and not query_implies_comparison(low_q)
            ):
                csv = supported_bank_keys_csv(self._bank_aliases)
                self._update_state(state, normalized, "", None, None)
                return RuntimeResponse(
                    answer_text=bank_clarification_message(csv),
                    status="clarify",
                    refusal_reason=None,
                    detected_topic=topic.label,
                    decision_trace=trace if req.verbose else [],
                )

        runtime_topic = topic.label
        retrieved = self._retriever.retrieve(
            RetrievalRequest(
                query=effective_query,
                index_name=req.index_name,
                top_k=req.top_k or self._default_top_k,
                topic=runtime_topic,  # type: ignore[arg-type]
                bank_keys=bank_keys,
            )
        )
        trace.append(f"retrieved_count={len(retrieved)}")
        distinct_retrieval_banks = _distinct_bank_keys_in_retrieval(retrieved)
        if self._orchestration.clarify_when_unscoped_multi_bank_evidence:
            if (
                bank_keys is None
                and not query_implies_all_banks(low_q)
                and not query_implies_comparison(low_q)
                and len(distinct_retrieval_banks) >= 2
            ):
                csv = supported_bank_keys_csv(self._bank_aliases)
                self._update_state(state, normalized, "", None, None)
                return RuntimeResponse(
                    answer_text=REFUSAL_RULES["clarify_multi_bank"].format(banks=csv),
                    status="clarify",
                    refusal_reason=None,
                    detected_topic=runtime_topic,
                    decision_trace=trace + ["clarify=unscoped_multi_bank_evidence"] if req.verbose else [],
                )
        if self._orchestration.restrict_evidence_to_single_bank_without_comparison:
            if (
                bank_keys is None
                and not query_implies_all_banks(low_q)
                and not query_implies_comparison(low_q)
                and retrieved
            ):
                dom = dominant_bank_key_from_chunks(retrieved)
                if dom:
                    retrieved = [c for c in retrieved if (c.chunk.bank_key or "").strip().lower() == dom]
                    trace.append(f"evidence_restricted_to_bank={dom}")

        evidence = self._evidence_checker.assess(effective_query, runtime_topic, retrieved)
        trace.append(f"evidence.sufficient={evidence.sufficient} reason={evidence.reason} max_score={evidence.max_score:.3f}")
        if not evidence.sufficient:
            self._update_state(state, normalized, "", bank_keys, runtime_topic)  # keep context
            det_one, det_list = _response_bank_fields(bank_keys)
            return RuntimeResponse(
                answer_text=refusal_message("insufficient_evidence"),
                status="refused",
                refusal_reason="insufficient_evidence",
                detected_topic=runtime_topic,
                detected_bank=det_one,
                detected_banks=det_list,
                retrieved_chunks_summary=[f"{c.chunk.chunk_id}:{c.score:.3f}" for c in retrieved[:3]],
                decision_trace=trace if req.verbose else [],
            )

        if self._orchestration.refuse_comparison_without_multi_bank_evidence:
            if query_implies_comparison(low_q):
                dpost = _distinct_bank_keys_in_retrieval(retrieved)
                if len(dpost) < 2:
                    self._update_state(state, normalized, "", bank_keys, runtime_topic)
                    det_one, det_list = _response_bank_fields(bank_keys)
                    trace.append("refuse=comparison_insufficient_banks_in_evidence")
                    return RuntimeResponse(
                        answer_text=refusal_message("comparison_insufficient"),
                        status="refused",
                        refusal_reason="comparison_insufficient",
                        detected_topic=runtime_topic,
                        detected_bank=det_one,
                        detected_banks=det_list,
                        retrieved_chunks_summary=[f"{c.chunk.chunk_id}:{c.score:.3f}" for c in retrieved[:3]],
                        decision_trace=trace if req.verbose else [],
                    )

        ctx_parts: list[str] = []
        for prev_u in state.recent_user_turns[-3:]:
            ctx_parts.append(f"Նախորդ հարց՝ {prev_u[:300]}")
        for prev_a in state.recent_assistant_turns[-2:]:
            ctx_parts.append(f"Նախորդ AI պատասխան (կրճատ)՝ {prev_a[:380]}")
        conv_context = "\n".join(ctx_parts) if ctx_parts else None

        prepared = prepare_evidence_for_answer(retrieved, max_chunks=self._max_evidence_chunks)
        if not prepared:
            prepared = retrieved[: self._max_evidence_chunks]

        if query_implies_comparison(low_q):
            answer_mode: AnswerMode = "comparison"
        elif bank_keys is not None and len(bank_keys) > 1:
            answer_mode = "multi_bank"
        elif query_implies_all_banks(low_q):
            answer_mode = "multi_bank"
        else:
            answer_mode = "single_bank"
        trace.append(f"answer_mode={answer_mode}")

        if isinstance(self._answer_generator, LLMAnswerGenerator):
            ar = self._answer_generator.generate_answer_result(
                effective_query,
                runtime_topic,
                prepared,
                bank_keys,
                context=conv_context,
                answer_mode=answer_mode,
            )
        else:
            text_only = self._answer_generator.generate(
                effective_query,
                runtime_topic,
                prepared,
                bank_keys,
                context=conv_context,
            )
            ar = AnswerResult(text=text_only, answer_synthesis="extractive_only", llm_error=None)

        self._update_state(state, normalized, ar.text, bank_keys, runtime_topic)
        det_one, det_list = _response_bank_fields(bank_keys)
        return RuntimeResponse(
            answer_text=ar.text,
            status="answered",
            answer_synthesis=ar.answer_synthesis,
            llm_error=ar.llm_error,
            detected_topic=runtime_topic,
            detected_bank=det_one,
            detected_banks=det_list,
            used_sources=dedupe_urls(
                [c.chunk.source_url for c in retrieved if getattr(c.chunk, "source_url", None)],
                max_n=8,
            ),
            retrieved_chunks_summary=[
                f"{c.chunk.bank_key}/{c.chunk.topic} {c.score:.3f} {c.chunk.section_title}" for c in retrieved[:3]
            ],
            state_updates={
                "last_topic": runtime_topic,
                "last_bank": det_one or "",
                "detected_banks": ",".join(det_list),
            },
            decision_trace=trace if req.verbose else [],
        )

    @staticmethod
    def _update_state(
        state: SessionState,
        user_turn: str,
        assistant_turn: str,
        bank_keys: frozenset[str] | None,
        topic: TopicLabel | None,
    ) -> None:
        state.recent_user_turns = (state.recent_user_turns + [user_turn])[-5:]
        if assistant_turn:
            state.recent_assistant_turns = (state.recent_assistant_turns + [assistant_turn])[-5:]
        if bank_keys is not None:
            state.last_bank_keys = sorted(bank_keys)
        else:
            state.last_bank_keys = []
        if bank_keys:
            if len(bank_keys) == 1:
                state.last_bank = next(iter(bank_keys))
            else:
                state.last_bank = None
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

