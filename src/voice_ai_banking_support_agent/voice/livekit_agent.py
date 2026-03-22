"""
Self-hosted LiveKit agent: push-to-talk data messages, mic PCM capture, STT → RAG → TTS.

Split modules: ``voice_topics`` (data-channel names), ``livekit_mic`` (remote track + consumer
lifecycle), ``livekit_playout`` (TTS decode, resample, frame pacing into ``AudioSource``).
"""

from __future__ import annotations

import array
import asyncio
import io
import json
import logging
import wave
from dataclasses import dataclass

from ..runtime.models import RuntimeResponse
from ..runtime.orchestrator import RuntimeOrchestrator, RuntimeRequest
from ..runtime.session_state import SessionStateStore
from .runtime_chat_client import RuntimeChatClient
from .session_handler import (
    build_runtime_session_id,
    resolve_runtime_session_id,
    sanitize_runtime_session_id_override,
)
from .hy_stt_postprocess import normalize_stt_transcript_hy
from .livekit_mic import (
    cancel_audio_consumer_task,
    find_remote_audio_track,
    wait_for_remote_audio_track,
)
from .stt import STTProvider, is_mock_stt_placeholder
from .tts import TTSProvider
from .tts_chunking import split_for_sequential_tts
from .tts_speech_prepare import prepare_text_for_tts
from .livekit_playout import publish_pcm_s16le_to_audio_source, tts_bytes_to_mono_s16le_at_rate
from .voice_config import VoiceConfig
from .voice_models import STTInput, VoiceTurnResult
from .voice_topics import (
    TOPIC_ASSISTANT_TEXT,
    TOPIC_ASSISTANT_TEXT_DELTA,
    TOPIC_PTT,
    TOPIC_VOICE_STATE,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
)
from .voice_turn_log import VoiceTurnLog

logger = logging.getLogger(__name__)


def _safe_next_chunk(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


@dataclass(frozen=True)
class LiveKitParticipantContext:
    room_name: str
    participant_identity: str


class LiveKitVoiceAgent:
    """
    Voice transport: STT → answers via FastAPI ``POST /chat`` (default, shared session with web UI)
    or in-process ``RuntimeOrchestrator`` (smoke / VOICE_RUNTIME_HTTP=0).

    Push-to-talk: the browser sends `voice.ptt` data packets (`start` / `end`) and only
    publishes mic audio while recording. The agent buffers PCM while `start` is active
    and runs STT → runtime → TTS once on `end`.
    """

    def __init__(
        self,
        *,
        runtime: RuntimeOrchestrator | None,
        state_store: SessionStateStore | None,
        stt_provider: STTProvider,
        tts_provider: TTSProvider,
        voice_config: VoiceConfig,
        chat_client: RuntimeChatClient | None = None,
    ) -> None:
        if chat_client is not None and (runtime is not None or state_store is not None):
            raise ValueError("Use either chat_client (HTTP /chat) or runtime+state_store, not both.")
        if chat_client is None and (runtime is None or state_store is None):
            raise ValueError("LiveKitVoiceAgent needs chat_client or both runtime and state_store.")
        self._runtime = runtime
        self._state_store = state_store
        self._chat_client = chat_client
        self._stt = stt_provider
        self._tts = tts_provider
        self._voice_config = voice_config
        self._turn_lock = asyncio.Lock()
        self._ptt_recording: dict[str, bool] = {}
        self._ptt_buffers: dict[str, list[bytes]] = {}
        self._audio_consumer_tasks: dict[str, asyncio.Task[None]] = {}
        self._ptt_finalize_tasks: dict[str, asyncio.Task[None]] = {}
        self._ptt_session_id_hint: dict[str, str] = {}
        # Latest RemoteAudioTrack per user (updated on subscribe); consumer restarted each PTT start.
        self._remote_mic_track_ref: dict[str, object] = {}
        self._ptt_turn_log: dict[str, VoiceTurnLog] = {}

    def process_turn(
        self,
        *,
        participant: LiveKitParticipantContext,
        payload: STTInput,
        index_name: str,
    ) -> VoiceTurnResult:
        if self._runtime is None or self._state_store is None:
            raise RuntimeError(
                "process_turn is for voice-smoke only and requires in-process runtime "
                "(run with VOICE_RUNTIME_HTTP=0 and chat_client unset)."
            )
        if payload.encoding == "text":
            user_text = payload.content.decode("utf-8", errors="ignore").strip()
        else:
            user_text = normalize_stt_transcript_hy(self._stt.transcribe(payload).strip())
            if not user_text:
                logger.warning("STT returned empty text; orchestrator may refuse or mis-route.")
            elif "[mock-stt-unavailable]" in user_text:
                logger.warning(
                    "STT returned mock placeholder (no real transcription). Set VOICE_STT_ENDPOINT in .env "
                    "for Armenian speech-to-text, or send JSON text via LiveKit data packets."
                )
        return self._run_runtime_and_synthesize(
            participant=participant, user_text=user_text, index_name=index_name
        )

    def _run_runtime_and_synthesize(
        self,
        *,
        participant: LiveKitParticipantContext,
        user_text: str,
        index_name: str,
    ) -> VoiceTurnResult:
        session_id = build_runtime_session_id(
            room_name=participant.room_name,
            participant_identity=participant.participant_identity,
        )
        state = self._state_store.get_or_create(session_id)
        runtime_response = self._runtime.handle(
            RuntimeRequest(
                session_id=session_id,
                query=user_text,
                index_name=index_name,
                top_k=self._voice_config.behavior.chat_top_k,
                verbose=self._voice_config.behavior.verbose_trace,
            ),
            state,
        )
        answer_text = runtime_response.answer_text[: self._voice_config.behavior.max_response_chars]
        tts_output = self._tts.synthesize(prepare_text_for_tts(answer_text))
        return VoiceTurnResult(
            session_id=session_id,
            user_text=user_text,
            runtime_response=runtime_response,
            tts_output=tts_output,
        )

    def run_self_hosted(self, *, index_name: str) -> None:
        asyncio.run(self._run_self_hosted_async(index_name=index_name))

    async def _publish_voice_state(self, room, *, state: str, detail: str | None = None) -> None:
        body = json.dumps({"state": state, "detail": detail}, ensure_ascii=False).encode("utf-8")
        await room.local_participant.publish_data(body, reliable=True, topic=TOPIC_VOICE_STATE)

    async def _run_self_hosted_async(self, *, index_name: str) -> None:
        try:
            import livekit.rtc as rtc  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LiveKit SDK not installed. Install `livekit` package to run live transport."
            ) from exc

        logger.info(
            "Starting self-hosted LiveKit voice agent room=%s url=%s",
            self._voice_config.livekit.room_name,
            self._voice_config.livekit.url,
        )
        room = rtc.Room()
        token = self._resolve_agent_token()
        ice = rtc.IceServer()
        ice.urls.append("stun:stun.l.google.com:19302")
        rtc_cfg = rtc.RtcConfiguration()
        rtc_cfg.ice_servers.append(ice)
        lk_connect_opts = rtc.RoomOptions(connect_timeout=60.0, rtc_config=rtc_cfg)
        try:
            await room.connect(self._voice_config.livekit.url, token, lk_connect_opts)
        except Exception as exc:
            logger.error(
                "LiveKit connect failed url=%s (check LIVEKIT_URL, token scope, and server).",
                self._voice_config.livekit.url,
            )
            raise RuntimeError(
                "Could not connect to LiveKit. Verify LIVEKIT_URL, LIVEKIT_TOKEN, and that the server is running."
            ) from exc

        pub_sr = max(8000, int(self._voice_config.behavior.livekit_publish_sample_rate))
        out_source = rtc.AudioSource(sample_rate=pub_sr, num_channels=1)
        out_track = rtc.LocalAudioTrack.create_audio_track("assistant-tts", out_source)
        await room.local_participant.publish_track(out_track)
        logger.info("Published assistant audio track.")

        agent_id = self._voice_config.livekit.agent_identity

        @room.on("track_subscribed")
        def _on_track_subscribed(track, publication, participant):  # pragma: no cover
            if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
                return
            if participant.identity == agent_id:
                return
            pid = participant.identity
            self._remote_mic_track_ref[pid] = track
            logger.info(
                "Remote mic track cached participant=%s (consumer starts on next PTT start).",
                pid,
            )

        @room.on("track_unsubscribed")
        def _on_track_unsubscribed(track, publication, participant):  # pragma: no cover
            if participant.identity == agent_id:
                return
            pid = participant.identity
            self._remote_mic_track_ref.pop(pid, None)
            task = self._audio_consumer_tasks.pop(pid, None)
            if task and not task.done():
                task.cancel()

        @room.on("data_received")
        def _on_data(packet) -> None:  # pragma: no cover
            asyncio.create_task(
                self._handle_data_received(
                    rtc=rtc,
                    room=room,
                    out_source=out_source,
                    packet=packet,
                    index_name=index_name,
                )
            )

        logger.info("LiveKit agent connected; push-to-talk on topic %s.", TOPIC_PTT)
        while True:  # pragma: no cover
            await asyncio.sleep(1)

    async def _consume_remote_audio_track(
        self,
        *,
        rtc,
        track,
        participant_identity: str,
    ) -> None:
        stream = rtc.AudioStream(track=track, sample_rate=16000, num_channels=1)
        try:
            async for evt in stream:  # pragma: no cover
                if not self._ptt_recording.get(participant_identity, False):
                    continue
                frame = evt.frame
                pcm = bytes(frame.data)
                self._ptt_buffers.setdefault(participant_identity, []).append(pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Audio consumer failed participant=%s", participant_identity)
        else:
            logger.warning(
                "Mic AudioStream ended participant=%s (unpublish or SDK close). "
                "Consumer will be recreated on the next PTT start.",
                participant_identity,
            )

    async def _ensure_mic_consumer(
        self,
        *,
        rtc,
        room,
        participant_identity: str,
        vlog: VoiceTurnLog,
    ) -> None:
        """Start a fresh audio consumer; required every turn after the previous stream ended."""
        vlog.event("mic_consumer_ensure_start")
        track = self._remote_mic_track_ref.get(participant_identity)
        if track is None:
            track = find_remote_audio_track(room, participant_identity=participant_identity)
        if track is None:
            track = await wait_for_remote_audio_track(
                room,
                participant_identity=participant_identity,
                max_wait_s=self._voice_config.behavior.mic_track_wait_seconds,
            )
        if track is None:
            vlog.event(
                "mic_consumer_no_track",
                wait_s=self._voice_config.behavior.mic_track_wait_seconds,
            )
            logger.warning(
                "No remote mic track for participant=%s after %.2fs — audio buffer may be empty.",
                participant_identity,
                self._voice_config.behavior.mic_track_wait_seconds,
            )
            return
        self._remote_mic_track_ref[participant_identity] = track
        prev = self._audio_consumer_tasks.pop(participant_identity, None)
        await cancel_audio_consumer_task(prev)
        self._audio_consumer_tasks[participant_identity] = asyncio.create_task(
            self._consume_remote_audio_track(
                rtc=rtc,
                track=track,
                participant_identity=participant_identity,
            )
        )
        vlog.event("mic_consumer_task_spawned")

    async def _handle_data_received(
        self,
        *,
        rtc,
        room,
        out_source,
        packet,
        index_name: str,
    ) -> None:
        topic = (getattr(packet, "topic", None) or "").strip()
        try:
            body = json.loads(bytes(packet.data).decode("utf-8"))
        except Exception:
            logger.warning("Ignoring non-JSON data packet topic=%s", topic)
            return

        if topic == TOPIC_PTT:
            await self._handle_ptt_packet(
                rtc=rtc,
                room=room,
                out_source=out_source,
                body=body,
                index_name=index_name,
                packet=packet,
            )
            return

        # Legacy: JSON text queries (no topic), for manual testing.
        if topic == "" and isinstance(body, dict) and "text" in body:
            await self._handle_legacy_text_packet(
                rtc=rtc,
                room=room,
                out_source=out_source,
                body=body,
                index_name=index_name,
            )

    def _resolve_participant_identity(self, packet, body: dict) -> str | None:
        if packet.participant is not None:
            return packet.participant.identity
        ident = body.get("participant_identity")
        return str(ident).strip() if ident else None

    async def _handle_ptt_packet(
        self,
        *,
        rtc,
        room,
        out_source,
        packet,
        body: dict,
        index_name: str,
    ) -> None:
        ptype = (body.get("type") or "").strip().lower()
        pid = self._resolve_participant_identity(packet, body)
        if not pid:
            logger.warning("PTT packet missing participant identity")
            return
        if pid == self._voice_config.livekit.agent_identity:
            return

        if ptype == "start":
            if self._turn_lock.locked():
                await self._publish_voice_state(room, state="busy", detail="assistant_turn_in_progress")
                return
            if self._ptt_recording.get(pid):
                logger.info("PTT start ignored (already recording) participant=%s", pid)
                await self._publish_voice_state(room, state="listening")
                return
            self._ptt_recording[pid] = True
            self._ptt_buffers[pid] = []
            sid_raw = body.get("session_id") or body.get("runtime_session_id")
            cleaned = None
            if isinstance(sid_raw, str):
                cleaned = sanitize_runtime_session_id_override(sid_raw)
            if cleaned:
                self._ptt_session_id_hint[pid] = cleaned
            tlog = VoiceTurnLog()
            self._ptt_turn_log[pid] = tlog
            tlog.event("ptt_record_start", participant=pid)
            await self._ensure_mic_consumer(
                rtc=rtc, room=room, participant_identity=pid, vlog=tlog
            )
            logger.info("PTT start participant=%s turn_id=%s", pid, tlog.turn_id)
            await self._publish_voice_state(room, state="listening")
            return

        if ptype == "end":
            prev = self._ptt_finalize_tasks.get(pid)
            if prev is not None and not prev.done():
                logger.info("PTT end ignored (finalize already running) participant=%s", pid)
                return

            sid_end = body.get("session_id") or body.get("runtime_session_id")
            sid_override = (
                sanitize_runtime_session_id_override(sid_end)
                if isinstance(sid_end, str)
                else None
            )
            if not sid_override:
                sid_override = self._ptt_session_id_hint.pop(pid, None)
            else:
                self._ptt_session_id_hint.pop(pid, None)

            async def _finalize_wrapper() -> None:
                try:
                    await self._finalize_ptt_turn(
                        rtc=rtc,
                        room=room,
                        out_source=out_source,
                        participant_identity=pid,
                        index_name=index_name,
                        runtime_session_id_override=sid_override,
                    )
                finally:
                    self._ptt_finalize_tasks.pop(pid, None)

            self._ptt_finalize_tasks[pid] = asyncio.create_task(_finalize_wrapper())
            return

        logger.warning("Unknown PTT type=%s", ptype)

    @staticmethod
    def _boost_quiet_pcm_s16le(pcm: bytes) -> bytes:
        """Whisper VAD may drop quiet mic audio; gently normalize sub-threshold captures."""
        if len(pcm) < 4 or len(pcm) % 2 != 0:
            return pcm
        samples = array.array("h")
        samples.frombytes(pcm)
        if not samples:
            return pcm
        peak = max(abs(x) for x in samples)
        if peak < 2000 and peak > 0:
            target = 20000
            scale = min(target / float(peak), 10.0)
            clipped = array.array(
                "h",
                (max(-32768, min(32767, int(round(x * scale)))) for x in samples),
            )
            return clipped.tobytes()
        return pcm

    async def _play_tts_answer(
        self,
        *,
        rtc,
        room,
        out_source,
        answer_text: str,
        participant_identity: str,
        vlog: VoiceTurnLog | None = None,
    ) -> bool:
        """Returns False if TTS/playout failed (caller publishes error; idle comes from turn finally)."""
        tchunks = split_for_sequential_tts(answer_text)
        if not tchunks:
            tchunks = [answer_text]
        chunk_i = -1
        tts_timeout = max(24.0, float(self._voice_config.tts.timeout_seconds))
        try:
            for chunk_i, chunk in enumerate(tchunks):
                piece = (chunk or "").strip()
                if not piece:
                    continue
                if vlog and chunk_i == 0:
                    vlog.event("tts_chunk_start", index=chunk_i, chars=len(piece))
                tts_output = await asyncio.wait_for(
                    asyncio.to_thread(self._tts.synthesize, piece),
                    timeout=tts_timeout,
                )
                await self._publish_tts_audio(
                    rtc=rtc,
                    out_source=out_source,
                    tts_audio=tts_output.audio,
                    tts_encoding=tts_output.encoding,
                )
        except asyncio.TimeoutError:
            logger.exception(
                "TTS timeout participant=%s chunk_index=%s", participant_identity, chunk_i
            )
            if vlog:
                vlog.event("tts_timeout", chunk_index=chunk_i)
            await self._publish_voice_state(room, state="error", detail="tts_failed")
            return False
        except Exception:
            logger.exception("TTS failed participant=%s chunk_index=%s", participant_identity, chunk_i)
            if vlog:
                vlog.fail("tts", RuntimeError("tts_failed"))
            await self._publish_voice_state(room, state="error", detail="tts_failed")
            return False
        if vlog:
            vlog.event("tts_playback_done", chunks=len(tchunks))
        return True

    async def _deliver_assistant_payload_and_tts(
        self,
        *,
        rtc,
        room,
        out_source,
        session_id: str,
        rr: RuntimeResponse,
        participant_identity: str,
        streamed: bool,
        vlog: VoiceTurnLog | None = None,
    ) -> None:
        max_c = self._voice_config.behavior.max_response_chars
        answer_text = (rr.answer_text or "")[:max_c]
        packet = json.dumps(
            {
                "session_id": session_id,
                "answer_text": answer_text,
                "status": rr.status,
                "refusal_reason": rr.refusal_reason,
                "answer_synthesis": rr.answer_synthesis,
                "llm_error": rr.llm_error,
                "decision_trace": rr.decision_trace,
                "tts_audio_bytes": 0,
                "streamed": streamed,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        await room.local_participant.publish_data(packet, reliable=True, topic=TOPIC_ASSISTANT_TEXT)
        await self._publish_voice_state(room, state="speaking")
        speak_text = prepare_text_for_tts(answer_text)
        await self._play_tts_answer(
            rtc=rtc,
            room=room,
            out_source=out_source,
            answer_text=speak_text,
            participant_identity=participant_identity,
            vlog=vlog,
        )

    async def _consume_runtime_stream(
        self,
        *,
        rtc,
        room,
        out_source,
        session_id: str,
        req: RuntimeRequest,
        state,
        participant_identity: str,
        vlog: VoiceTurnLog | None = None,
    ):
        """
        Run ``stream_handle`` (Gemini token streaming), push deltas to the UI, then play TTS
        on the final scrubbed answer (same grounding as non-streaming ``handle``).
        """

        if self._runtime is None:
            raise RuntimeError("stream_handle requires in-process runtime (set VOICE_RUNTIME_HTTP=0).")
        gen = self._runtime.stream_handle(req, state)
        final = None
        stream_chunk_timeout = max(90.0, float(self._voice_config.behavior.runtime_api_timeout_seconds))
        try:
            while True:
                chunk = await asyncio.wait_for(
                    asyncio.to_thread(_safe_next_chunk, gen),
                    timeout=stream_chunk_timeout,
                )
                if chunk is None:
                    break
                if chunk.text_delta:
                    dp = json.dumps({"text": chunk.text_delta}, ensure_ascii=False).encode("utf-8")
                    await room.local_participant.publish_data(
                        dp, reliable=True, topic=TOPIC_ASSISTANT_TEXT_DELTA
                    )
                if chunk.done is not None:
                    final = chunk.done
                    break
        finally:
            close = getattr(gen, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("stream_handle generator close", exc_info=True)
        if final is None:
            raise RuntimeError("stream_handle ended without a terminal response")

        answer_text = final.answer_text[: self._voice_config.behavior.max_response_chars]
        packet = json.dumps(
            {
                "session_id": session_id,
                "answer_text": answer_text,
                "status": final.status,
                "refusal_reason": final.refusal_reason,
                "answer_synthesis": final.answer_synthesis,
                "llm_error": final.llm_error,
                "decision_trace": final.decision_trace,
                "tts_audio_bytes": 0,
                "streamed": True,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        await room.local_participant.publish_data(packet, reliable=True, topic=TOPIC_ASSISTANT_TEXT)

        await self._publish_voice_state(room, state="speaking")
        speak_text = prepare_text_for_tts(answer_text)
        if not await self._play_tts_answer(
            rtc=rtc,
            room=room,
            out_source=out_source,
            answer_text=speak_text,
            participant_identity=participant_identity,
            vlog=vlog,
        ):
            return final

        return final

    async def _finalize_ptt_turn(
        self,
        *,
        rtc,
        room,
        out_source,
        participant_identity: str,
        index_name: str,
        runtime_session_id_override: str | None = None,
    ) -> None:
        """
        One PTT turn: drain mic buffer → STT → RAG/LLM → TTS.

        **Root cause (second-turn hang / no audio):** the mic ``AudioStream`` consumer runs once per
        track subscription; after unpublish/republish or SDK stream end the task exits. A fresh
        consumer is started on each ``PTT start`` via ``_ensure_mic_consumer``. This method always
        ends in ``voice.state=idle`` in ``finally`` so the next turn cannot stay stuck in processing.
        """
        vlog = self._ptt_turn_log.pop(participant_identity, None) or VoiceTurnLog()
        stt_timeout = max(20.0, float(self._voice_config.stt.timeout_seconds))
        rt_timeout = max(30.0, float(self._voice_config.behavior.runtime_api_timeout_seconds))
        stream_turn_timeout = rt_timeout + 120.0

        async with self._turn_lock:
            vlog.event("finalize_enter", participant=participant_identity)
            try:
                # Keep _ptt_recording True during the trail pause so the mic consumer still appends
                # frames while the browser finishes sending audio (client waits ~220ms after PTT end
                # before unpublish). Stopping recording *before* this pause drops the tail and often
                # yields empty buffers on the second push-to-talk.
                vlog.event("record_stop")
                await asyncio.sleep(float(self._voice_config.behavior.pcm_trail_pause_seconds))
                self._ptt_recording[participant_identity] = False
                chunks = self._ptt_buffers.pop(participant_identity, [])
                pcm_blob = b"".join(chunks)
                await self._publish_voice_state(room, state="processing")

                min_bytes = 3200  # ~100ms mono int16@16k
                if len(pcm_blob) < min_bytes:
                    logger.warning(
                        "PTT end: audio too short participant=%s bytes=%s (min=%s). "
                        "Often fixed by VOICE_PCM_TRAIL_PAUSE_SECONDS >= 0.28 or speaking longer; "
                        "see README voice troubleshooting.",
                        participant_identity,
                        len(pcm_blob),
                        min_bytes,
                    )
                    vlog.event("audio_too_short", bytes=len(pcm_blob), min_bytes=min_bytes)
                    await self._publish_voice_state(room, state="error", detail="audio_too_short")
                    return

                pcm_blob = self._boost_quiet_pcm_s16le(pcm_blob)
                wav_bytes = self._pcm_to_wav(pcm_blob, sample_rate=16000, channels=1)
                participant = LiveKitParticipantContext(
                    room_name=self._voice_config.livekit.room_name,
                    participant_identity=participant_identity,
                )
                await self._publish_voice_state(room, state="processing", detail="transcribing")
                vlog.event("stt_start")
                try:
                    user_text = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._stt.transcribe,
                            STTInput(
                                content=wav_bytes,
                                encoding="wav",
                                language=self._voice_config.stt.language,
                            ),
                        ),
                        timeout=stt_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error("STT timeout participant=%s", participant_identity)
                    vlog.event("stt_timeout")
                    await self._publish_voice_state(room, state="error", detail="stt_failed")
                    return
                except Exception as exc:
                    logger.exception("STT failed participant=%s", participant_identity)
                    vlog.fail("stt", exc)
                    await self._publish_voice_state(room, state="error", detail="stt_failed")
                    return

                vlog.event("stt_end")

                if is_mock_stt_placeholder(user_text):
                    logger.error(
                        "STT mock placeholder — configure VOICE_STT_ENDPOINT (e.g. :8088/transcribe)."
                    )
                    vlog.event("stt_mock_placeholder")
                    await self._publish_voice_state(room, state="error", detail="stt_service_missing")
                    return

                user_text = normalize_stt_transcript_hy((user_text or "").strip())
                if not user_text:
                    logger.warning("STT empty transcript participant=%s", participant_identity)
                    vlog.event("stt_empty_result")
                    await self._publish_voice_state(room, state="error", detail="stt_empty")
                    return

                logger.info(
                    "PTT STT ok turn_id=%s participant=%s preview=%r",
                    vlog.turn_id,
                    participant_identity,
                    user_text[:240] + ("…" if len(user_text) > 240 else ""),
                )

                tr_packet = json.dumps(
                    {
                        "text": user_text,
                        "final": True,
                        "participant_identity": participant_identity,
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                await room.local_participant.publish_data(
                    tr_packet, reliable=True, topic=TOPIC_VOICE_TRANSCRIPT_FINAL
                )

                await self._publish_voice_state(room, state="processing", detail="answering")
                session_id = resolve_runtime_session_id(
                    room_name=participant.room_name,
                    participant_identity=participant.participant_identity,
                    override=runtime_session_id_override,
                )
                req = RuntimeRequest(
                    session_id=session_id,
                    query=user_text,
                    index_name=index_name,
                    top_k=self._voice_config.behavior.chat_top_k,
                    verbose=self._voice_config.behavior.verbose_trace,
                )
                vlog.event("llm_start", session_id=session_id)
                if self._chat_client is not None:
                    try:
                        rr = await asyncio.wait_for(
                            asyncio.to_thread(self._chat_client.chat, req),
                            timeout=rt_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.error("Runtime /chat timeout participant=%s", participant_identity)
                        vlog.event("llm_timeout")
                        await self._publish_voice_state(
                            room, state="error", detail="processing_failed"
                        )
                        return
                    except Exception as exc:
                        logger.exception("Runtime /chat failed participant=%s", participant_identity)
                        vlog.fail("llm", exc)
                        await self._publish_voice_state(
                            room, state="error", detail="processing_failed"
                        )
                        return
                    vlog.event("llm_end")
                    await self._deliver_assistant_payload_and_tts(
                        rtc=rtc,
                        room=room,
                        out_source=out_source,
                        session_id=session_id,
                        rr=rr,
                        participant_identity=participant_identity,
                        streamed=False,
                        vlog=vlog,
                    )
                elif self._voice_config.behavior.stream_llm_tokens:
                    assert self._runtime is not None and self._state_store is not None
                    state = self._state_store.get_or_create(session_id)
                    try:
                        await asyncio.wait_for(
                            self._consume_runtime_stream(
                                rtc=rtc,
                                room=room,
                                out_source=out_source,
                                session_id=session_id,
                                req=req,
                                state=state,
                                participant_identity=participant_identity,
                                vlog=vlog,
                            ),
                            timeout=stream_turn_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.error("Stream LLM/TTS timeout participant=%s", participant_identity)
                        vlog.event("stream_turn_timeout")
                        await self._publish_voice_state(
                            room, state="error", detail="processing_failed"
                        )
                        return
                    except Exception as exc:
                        logger.exception("Stream path failed participant=%s", participant_identity)
                        vlog.fail("stream", exc)
                        await self._publish_voice_state(
                            room, state="error", detail="processing_failed"
                        )
                        return
                else:
                    assert self._runtime is not None and self._state_store is not None
                    state = self._state_store.get_or_create(session_id)
                    try:
                        runtime_response = await asyncio.wait_for(
                            asyncio.to_thread(self._runtime.handle, req, state),
                            timeout=rt_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.error("Runtime handle timeout participant=%s", participant_identity)
                        vlog.event("llm_timeout")
                        await self._publish_voice_state(
                            room, state="error", detail="processing_failed"
                        )
                        return
                    except Exception as exc:
                        logger.exception("Runtime handle failed participant=%s", participant_identity)
                        vlog.fail("llm", exc)
                        await self._publish_voice_state(
                            room, state="error", detail="processing_failed"
                        )
                        return
                    await self._deliver_assistant_payload_and_tts(
                        rtc=rtc,
                        room=room,
                        out_source=out_source,
                        session_id=session_id,
                        rr=runtime_response,
                        participant_identity=participant_identity,
                        streamed=False,
                        vlog=vlog,
                    )
                vlog.event("turn_success")
            except asyncio.CancelledError:
                vlog.event("finalize_cancelled")
                raise
            except Exception as exc:
                logger.exception("Voice turn failed participant=%s", participant_identity)
                vlog.fail("turn", exc)
                await self._publish_voice_state(room, state="error", detail="processing_failed")
            finally:
                self._ptt_recording[participant_identity] = False
                vlog.event("cleanup_to_idle")
                await self._publish_voice_state(room, state="idle")
                logger.info(
                    "PTT finalize done turn_id=%s participant=%s",
                    vlog.turn_id,
                    participant_identity,
                )

    async def _handle_legacy_text_packet(
        self, *, rtc, room, out_source, body: dict, index_name: str
    ) -> None:
        async with self._turn_lock:
            await self._publish_voice_state(room, state="processing")
            participant_id = str(body.get("participant_identity") or "unknown")
            try:
                text = normalize_stt_transcript_hy(str(body.get("text", "") or "").strip())
                participant = LiveKitParticipantContext(
                    room_name=self._voice_config.livekit.room_name,
                    participant_identity=participant_id,
                )
                tr_packet = json.dumps(
                    {"text": text, "final": True, "participant_identity": participant_id},
                    ensure_ascii=False,
                ).encode("utf-8")
                await room.local_participant.publish_data(
                    tr_packet, reliable=True, topic=TOPIC_VOICE_TRANSCRIPT_FINAL
                )
                raw_sid = body.get("session_id") or body.get("runtime_session_id")
                sid_ov = (
                    sanitize_runtime_session_id_override(raw_sid)
                    if isinstance(raw_sid, str)
                    else None
                )
                session_id = resolve_runtime_session_id(
                    room_name=participant.room_name,
                    participant_identity=participant.participant_identity,
                    override=sid_ov,
                )
                lreq = RuntimeRequest(
                    session_id=session_id,
                    query=text,
                    index_name=index_name,
                    top_k=self._voice_config.behavior.chat_top_k,
                    verbose=self._voice_config.behavior.verbose_trace,
                )
                try:
                    if self._chat_client is not None:
                        runtime_response = await asyncio.wait_for(
                            asyncio.to_thread(self._chat_client.chat, lreq),
                            timeout=max(
                                30.0,
                                float(self._voice_config.behavior.runtime_api_timeout_seconds),
                            ),
                        )
                    else:
                        assert self._runtime is not None and self._state_store is not None
                        state = self._state_store.get_or_create(session_id)
                        runtime_response = await asyncio.wait_for(
                            asyncio.to_thread(self._runtime.handle, lreq, state),
                            timeout=max(
                                30.0,
                                float(self._voice_config.behavior.runtime_api_timeout_seconds),
                            ),
                        )
                except Exception:
                    logger.exception("Legacy data packet processing failed")
                    await self._publish_voice_state(room, state="error", detail="processing_failed")
                    return

                await self._deliver_assistant_payload_and_tts(
                    rtc=rtc,
                    room=room,
                    out_source=out_source,
                    session_id=session_id,
                    rr=runtime_response,
                    participant_identity=participant_id,
                    streamed=False,
                )
            finally:
                await self._publish_voice_state(room, state="idle")

    async def _publish_tts_audio(self, *, rtc, out_source, tts_audio: bytes, tts_encoding: str) -> None:
        try:
            target_sr = max(8000, int(self._voice_config.behavior.livekit_publish_sample_rate))
            pcm_mono = tts_bytes_to_mono_s16le_at_rate(
                tts_audio,
                encoding=tts_encoding,
                target_sample_rate=target_sr,
                pcm_assumed_rate_if_raw=int(self._voice_config.tts.pcm_s16le_sample_rate),
            )
            await publish_pcm_s16le_to_audio_source(
                rtc,
                out_source,
                pcm_mono,
                sample_rate=target_sr,
                num_channels=1,
                frame_ms=float(self._voice_config.behavior.livekit_playout_frame_ms),
                pace_realtime=bool(self._voice_config.behavior.livekit_playout_realtime_pacing),
            )
        except Exception:
            logger.exception("Failed to publish TTS audio track")

    @staticmethod
    def _pcm_to_wav(pcm: bytes, *, sample_rate: int, channels: int) -> bytes:
        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return out.getvalue()

    def _resolve_agent_token(self) -> str:
        """
        Join token: prefer LIVEKIT_TOKEN; else mint JWT from api_key/api_secret (same as HTTP /api/livekit/token).
        """

        import os

        from ..runtime.livekit_tokens import mint_participant_token

        raw = (os.getenv("LIVEKIT_TOKEN") or "").strip()
        if raw:
            return raw
        key = (self._voice_config.livekit.api_key or "").strip()
        secret = (self._voice_config.livekit.api_secret or "").strip()
        if key and secret:
            return mint_participant_token(
                identity=self._voice_config.livekit.agent_identity,
                room=self._voice_config.livekit.room_name,
                api_key=key,
                api_secret=secret,
            )
        raise RuntimeError(
            "LiveKit agent token missing. Set LIVEKIT_TOKEN, or LIVEKIT_API_KEY + LIVEKIT_API_SECRET "
            "(and optional LIVEKIT_ROOM) so the agent can mint a JWT like the browser client."
        )
