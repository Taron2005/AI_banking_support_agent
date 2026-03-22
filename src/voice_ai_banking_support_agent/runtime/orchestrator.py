from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterator

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
from .evidence_select import dedupe_urls, filter_chunks_to_bank_keys
from .followup_resolver import FollowUpResolver
from .models import RuntimeResponse, SessionState, TopicClassification
from .query_answer_hints import extra_llm_context, retrieval_query_with_topic_boost
from .query_normalizer import normalize_query, repair_stt_transcript
from .orchestration_policy import dominant_bank_key_from_chunks, supported_bank_keys_csv
from .rag_prompts import REFUSAL_RULES
from .refusal import bank_clarification_message, refusal_message
from .retriever import RetrievalRequest, RuntimeRetriever
from .runtime_config import OrchestrationSettings
from .topic_classifier import TopicClassifier

logger = logging.getLogger(__name__)

_TOPIC_LABELS = frozenset({"credit", "deposit", "branch"})

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


@dataclass(frozen=True)
class _PendingLlmTurn:
    """Internal: RAG + gates passed; ready for LLM or extractive answer synthesis."""

    normalized: str
    effective_query: str
    runtime_topic: str
    prepared: list
    bank_keys: frozenset[str] | None
    conv_context: str | None
    answer_mode: AnswerMode
    trace: list[str]
    retrieved: list
    verbose: bool


@dataclass(frozen=True)
class RuntimeStreamChunk:
    """Streaming turn: token deltas from Gemini, then a single terminal ``done`` response."""

    text_delta: str | None = None
    done: RuntimeResponse | None = None


class RuntimeOrchestrator:
    """Main text-runtime orchestration pipeline."""

    @staticmethod
    def _clear_pending_clarification(state: SessionState) -> None:
        state.pending_clarify = None
        state.pending_query = None
        state.pending_topic = None

    def _consume_pending_clarification(
        self, normalized: str, state: SessionState
    ) -> tuple[str, TopicLabel | None]:
        """
        If we were waiting for bank/scope after a clarify message, merge the user's reply
        with the stored question. Returns (query_for_pipeline, forced_topic_or_none).
        """

        if not state.pending_clarify or not state.pending_query:
            return normalized, None
        if self._followup_resolver.should_abort_pending_clarification(normalized):
            self._clear_pending_clarification(state)
            return normalized, None
        lower = normalized.lower()
        if state.pending_clarify in ("bank", "multi_bank"):
            if query_implies_all_banks(lower):
                pq = state.pending_query
                ft = state.pending_topic
                merged = f"{pq} {normalized}".strip()
                self._clear_pending_clarification(state)
                return merged, ft
            detected = self._bank_detector.detect_all(normalized)
            if len(detected) >= 1:
                merged = f"{state.pending_query} {normalized}".strip()
                ft = state.pending_topic
                self._clear_pending_clarification(state)
                return merged, ft
        return normalized, None

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

    def _dispatch_until_llm(self, req: RuntimeRequest, state: SessionState) -> RuntimeResponse | _PendingLlmTurn:
        trace: list[str] = []
        normalized = repair_stt_transcript(req.query)
        trace.append(f"normalized={normalized}")
        working_query, forced_topic_from_pending = self._consume_pending_clarification(normalized, state)
        if working_query != normalized:
            trace.append(f"pending_clarify_merged={working_query[:200]!r}")
        followup = self._followup_resolver.resolve(working_query, state)
        effective_query = followup.resolved_query
        trace.append(f"followup.used={followup.used_followup_context} merged={','.join(followup.merged_fields)}")
        if followup.needs_clarification:
            hint = (
                "Խնդրում եմ նախորդ հարցից հետո կրկնել հարցը ավելի կոնկրետ՝ նշելով բանկը և թեման "
                "(վարկ, ավանդ կամ մասնաճյուղ)։"
            )
            self._update_state(state, normalized, hint, None, None)
            return RuntimeResponse(
                answer_text=hint,
                status="clarify",
                refusal_reason=None,
                detected_topic=None,
                decision_trace=trace if req.verbose else [],
            )
        topic = self._topic_classifier.classify(effective_query)
        if (
            forced_topic_from_pending
            and forced_topic_from_pending in _TOPIC_LABELS
            and topic.label == "ambiguous"
        ):
            topic = TopicClassification(
                label=forced_topic_from_pending,  # type: ignore[arg-type]
                confidence=0.9,
                reason="carried_from_pending_bank_clarify",
                matched_terms=[],
            )
            trace.append(f"topic_overridden_from_pending={topic.label}")
        trace.append(f"topic={topic.label} conf={topic.confidence:.2f} reason={topic.reason}")

        if topic.label in ("out_of_scope", "unsupported_request_type", "ambiguous"):
            reason = (
                "unsupported_request_type"
                if topic.label == "unsupported_request_type"
                else ("ambiguous" if topic.label == "ambiguous" else "out_of_scope")
            )
            self._clear_pending_clarification(state)
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
                clarify_text = bank_clarification_message(csv)
                state.pending_clarify = "bank"
                state.pending_query = effective_query
                state.pending_topic = topic.label
                self._update_state(state, normalized, clarify_text, None, topic.label)
                return RuntimeResponse(
                    answer_text=clarify_text,
                    status="clarify",
                    refusal_reason=None,
                    detected_topic=topic.label,
                    decision_trace=trace if req.verbose else [],
                )

        runtime_topic = topic.label
        retrieval_query = retrieval_query_with_topic_boost(
            effective_query, runtime_topic=str(runtime_topic)
        )
        if retrieval_query != effective_query:
            trace.append(f"retrieval_query_boost={retrieval_query[:120]!r}")
        retrieved = self._retriever.retrieve(
            RetrievalRequest(
                query=retrieval_query,
                index_name=req.index_name,
                top_k=req.top_k or self._default_top_k,
                topic=runtime_topic,  # type: ignore[arg-type]
                bank_keys=bank_keys,
            )
        )
        trace.append(f"retrieved_count={len(retrieved)}")
        if bank_keys:
            before_scope = len(retrieved)
            retrieved = filter_chunks_to_bank_keys(retrieved, bank_keys)
            if before_scope != len(retrieved):
                trace.append(f"post_filter_bank_scope dropped={before_scope - len(retrieved)}")
        distinct_retrieval_banks = _distinct_bank_keys_in_retrieval(retrieved)
        if self._orchestration.clarify_when_unscoped_multi_bank_evidence:
            if (
                bank_keys is None
                and not query_implies_all_banks(low_q)
                and not query_implies_comparison(low_q)
                and len(distinct_retrieval_banks) >= 2
            ):
                csv = supported_bank_keys_csv(self._bank_aliases)
                clarify_text = REFUSAL_RULES["clarify_multi_bank"].format(banks=csv)
                state.pending_clarify = "multi_bank"
                state.pending_query = effective_query
                state.pending_topic = topic.label
                self._update_state(state, normalized, clarify_text, None, topic.label)
                return RuntimeResponse(
                    answer_text=clarify_text,
                    status="clarify",
                    refusal_reason=None,
                    detected_topic=topic.label,
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
            self._clear_pending_clarification(state)
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
                    self._clear_pending_clarification(state)
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
        low_eff = effective_query.lower()
        hint = extra_llm_context(effective_query, runtime_topic=str(runtime_topic))
        if hint:
            ctx_parts.append(hint)
        detail_markers = (
            "մանրամասն",
            "ավելի",
            "բացատրիր",
            "լայնածավալ",
            "ամբողջությամբ",
            "explain",
            "more detail",
            "tell me more",
        )
        if any(m in low_eff for m in detail_markers):
            ctx_parts.append(
                "Օգտատերը խնդրում է մանրամասն, կառուցվածքավոր պատասխան՝ ապացույցում երևացող փաստերի սահմաններում "
                "(առանց արտաքին գիտելիքի)․ չսահմանափակես քեզ 2–3 նախադասությամբ, եթե ապացույցը հարստ է։"
            )
        if followup.used_followup_context or working_query != normalized:
            ctx_parts.append(
                "Նշում․ օգտատիրոջ վերջին հաղորդագրությունը կարող է լինել կարճ (օր. միայն բանկի անուն)՝ "
                "լրացնելով նախորդ հարցը․ պատասխանի համակցված նշանակությունը, ոչ թե միայն կարճ տողը։"
            )
        for prev_u in state.recent_user_turns[-5:]:
            ctx_parts.append(f"Նախորդ հարց՝ {prev_u[:520]}")
        for prev_a in state.recent_assistant_turns[-4:]:
            ctx_parts.append(f"Նախորդ AI պատասխան (կրճատ)՝ {prev_a[:900]}")
        conv_context = "\n".join(ctx_parts) if ctx_parts else None

        prepared = prepare_evidence_for_answer(retrieved, max_chunks=self._max_evidence_chunks)
        if not prepared:
            prepared = retrieved[: self._max_evidence_chunks]
        prepared = filter_chunks_to_bank_keys(prepared, bank_keys)

        if query_implies_comparison(low_q):
            answer_mode: AnswerMode = "comparison"
        elif bank_keys is not None and len(bank_keys) > 1:
            answer_mode = "multi_bank"
        elif query_implies_all_banks(low_q):
            answer_mode = "multi_bank"
        else:
            answer_mode = "single_bank"
        trace.append(f"answer_mode={answer_mode}")

        return _PendingLlmTurn(
            normalized=normalized,
            effective_query=effective_query,
            runtime_topic=runtime_topic,
            prepared=prepared,
            bank_keys=bank_keys,
            conv_context=conv_context,
            answer_mode=answer_mode,
            trace=trace,
            retrieved=retrieved,
            verbose=req.verbose,
        )

    def _build_answered_response(
        self, pre: _PendingLlmTurn, ar: AnswerResult, state: SessionState
    ) -> RuntimeResponse:
        self._clear_pending_clarification(state)
        self._update_state(state, pre.normalized, ar.text, pre.bank_keys, pre.runtime_topic)
        det_one, det_list = _response_bank_fields(pre.bank_keys)
        return RuntimeResponse(
            answer_text=ar.text,
            status="answered",
            answer_synthesis=ar.answer_synthesis,
            llm_error=ar.llm_error,
            detected_topic=pre.runtime_topic,
            detected_bank=det_one,
            detected_banks=det_list,
            used_sources=dedupe_urls(
                [
                    c.chunk.source_url
                    for c in filter_chunks_to_bank_keys(pre.prepared, pre.bank_keys)
                    if getattr(c.chunk, "source_url", None)
                ],
                max_n=8,
            ),
            retrieved_chunks_summary=[
                f"{c.chunk.bank_key}/{c.chunk.topic} {c.score:.3f} {c.chunk.section_title}"
                for c in pre.retrieved[:3]
            ],
            state_updates={
                "last_topic": pre.runtime_topic,
                "last_bank": det_one or "",
                "detected_banks": ",".join(det_list),
            },
            decision_trace=pre.trace if pre.verbose else [],
        )

    def handle(self, req: RuntimeRequest, state: SessionState) -> RuntimeResponse:
        pre = self._dispatch_until_llm(req, state)
        if isinstance(pre, RuntimeResponse):
            return pre
        if isinstance(self._answer_generator, LLMAnswerGenerator):
            ar = self._answer_generator.generate_answer_result(
                pre.effective_query,
                pre.runtime_topic,
                pre.prepared,
                pre.bank_keys,
                context=pre.conv_context,
                answer_mode=pre.answer_mode,
            )
        else:
            text_only = self._answer_generator.generate(
                pre.effective_query,
                pre.runtime_topic,
                pre.prepared,
                pre.bank_keys,
                context=pre.conv_context,
            )
            ar = AnswerResult(text=text_only, answer_synthesis="extractive_only", llm_error=None)
        return self._build_answered_response(pre, ar, state)

    def stream_handle(self, req: RuntimeRequest, state: SessionState) -> Iterator[RuntimeStreamChunk]:
        pre = self._dispatch_until_llm(req, state)
        if isinstance(pre, RuntimeResponse):
            yield RuntimeStreamChunk(done=pre)
            return
        if not isinstance(self._answer_generator, LLMAnswerGenerator):
            text_only = self._answer_generator.generate(
                pre.effective_query,
                pre.runtime_topic,
                pre.prepared,
                pre.bank_keys,
                context=pre.conv_context,
            )
            ar = AnswerResult(text=text_only, answer_synthesis="extractive_only", llm_error=None)
            yield RuntimeStreamChunk(done=self._build_answered_response(pre, ar, state))
            return
        for piece in self._answer_generator.generate_answer_result_stream(
            pre.effective_query,
            pre.runtime_topic,
            pre.prepared,
            pre.bank_keys,
            context=pre.conv_context,
            answer_mode=pre.answer_mode,
        ):
            if piece.delta:
                yield RuntimeStreamChunk(text_delta=piece.delta)
            if piece.result is not None:
                yield RuntimeStreamChunk(done=self._build_answered_response(pre, piece.result, state))
                return

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

