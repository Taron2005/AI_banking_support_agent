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
from .stt import STTProvider, is_mock_stt_placeholder
from .tts import TTSProvider
from .tts_chunking import split_for_sequential_tts
from .voice_config import VoiceConfig
from .voice_models import STTInput, VoiceTurnResult

logger = logging.getLogger(__name__)

TOPIC_PTT = "voice.ptt"
TOPIC_ASSISTANT_TEXT = "assistant.text"
TOPIC_ASSISTANT_TEXT_DELTA = "assistant.text.delta"
TOPIC_VOICE_STATE = "voice.state"
TOPIC_VOICE_TRANSCRIPT_FINAL = "voice.transcript.final"


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
                verbose=self._voice_config.behavior.verbose_trace,
            ),
            state,
        )
        answer_text = runtime_response.answer_text[: self._voice_config.behavior.max_response_chars]
        tts_output = self._tts.synthesize(answer_text)
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

        out_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
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
            prev = self._audio_consumer_tasks.pop(pid, None)
            if prev and not prev.done():
                prev.cancel()
            self._audio_consumer_tasks[pid] = asyncio.create_task(
                self._consume_remote_audio_track(
                    rtc=rtc,
                    track=track,
                    participant_identity=pid,
                )
            )

        @room.on("track_unsubscribed")
        def _on_track_unsubscribed(track, publication, participant):  # pragma: no cover
            if participant.identity == agent_id:
                return
            task = self._audio_consumer_tasks.pop(participant.identity, None)
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
                packet=packet,
                body=body,
                index_name=index_name,
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
            logger.info("PTT start participant=%s", pid)
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
    ) -> bool:
        """Returns False if TTS/playout failed (caller should set voice error state)."""
        tchunks = split_for_sequential_tts(answer_text)
        if not tchunks:
            tchunks = [answer_text]
        chunk_i = -1
        try:
            for chunk_i, chunk in enumerate(tchunks):
                piece = (chunk or "").strip()
                if not piece:
                    continue
                tts_output = await asyncio.to_thread(self._tts.synthesize, piece)
                await self._publish_tts_audio(
                    rtc=rtc,
                    out_source=out_source,
                    tts_audio=tts_output.audio,
                    tts_encoding=tts_output.encoding,
                )
        except Exception:
            logger.exception("TTS failed participant=%s chunk_index=%s", participant_identity, chunk_i)
            await self._publish_voice_state(room, state="error", detail="tts_failed")
            await self._publish_voice_state(room, state="idle")
            return False
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
        await self._play_tts_answer(
            rtc=rtc,
            room=room,
            out_source=out_source,
            answer_text=answer_text,
            participant_identity=participant_identity,
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
    ):
        """
        Run ``stream_handle`` (Gemini token streaming), push deltas to the UI, then play TTS
        on the final scrubbed answer (same grounding as non-streaming ``handle``).
        """

        if self._runtime is None:
            raise RuntimeError("stream_handle requires in-process runtime (set VOICE_RUNTIME_HTTP=0).")
        gen = self._runtime.stream_handle(req, state)
        final = None
        while True:
            chunk = await asyncio.to_thread(_safe_next_chunk, gen)
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
        if not await self._play_tts_answer(
            rtc=rtc,
            room=room,
            out_source=out_source,
            answer_text=answer_text,
            participant_identity=participant_identity,
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
        async with self._turn_lock:
            self._ptt_recording[participant_identity] = False
            await asyncio.sleep(0.25)
            chunks = self._ptt_buffers.pop(participant_identity, [])
            pcm_blob = b"".join(chunks)
            await self._publish_voice_state(room, state="processing")

            min_bytes = 3200  # ~100ms mono int16@16k
            if len(pcm_blob) < min_bytes:
                logger.info(
                    "PTT end: audio too short participant=%s bytes=%s (duplicate end or empty buffer)",
                    participant_identity,
                    len(pcm_blob),
                )
                await self._publish_voice_state(room, state="idle")
                return

            pcm_blob = self._boost_quiet_pcm_s16le(pcm_blob)
            wav_bytes = self._pcm_to_wav(pcm_blob, sample_rate=16000, channels=1)
            participant = LiveKitParticipantContext(
                room_name=self._voice_config.livekit.room_name,
                participant_identity=participant_identity,
            )
            await self._publish_voice_state(room, state="processing", detail="transcribing")
            try:
                user_text = await asyncio.to_thread(
                    self._stt.transcribe,
                    STTInput(
                        content=wav_bytes,
                        encoding="wav",
                        language=self._voice_config.stt.language,
                    ),
                )
            except Exception:
                logger.exception("STT failed participant=%s", participant_identity)
                await self._publish_voice_state(room, state="error", detail="stt_failed")
                await self._publish_voice_state(room, state="idle")
                return

            if is_mock_stt_placeholder(user_text):
                logger.error(
                    "STT returned mock placeholder — start scripts/voice_http_stt_server.py and set "
                    "VOICE_STT_ENDPOINT=http://127.0.0.1:8088/transcribe in .env (or fix HTTP STT errors)."
                )
                await self._publish_voice_state(room, state="error", detail="stt_service_missing")
                await self._publish_voice_state(room, state="idle")
                return

            user_text = normalize_stt_transcript_hy((user_text or "").strip())
            if not user_text:
                logger.warning("STT returned empty transcript participant=%s", participant_identity)
                await self._publish_voice_state(room, state="error", detail="stt_empty")
                await self._publish_voice_state(room, state="idle")
                return

            logger.info(
                "PTT STT ok participant=%s chars=%s preview=%r",
                participant_identity,
                len(user_text),
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
                verbose=self._voice_config.behavior.verbose_trace,
            )
            try:
                if self._chat_client is not None:
                    rr = await asyncio.to_thread(self._chat_client.chat, req)
                    await self._deliver_assistant_payload_and_tts(
                        rtc=rtc,
                        room=room,
                        out_source=out_source,
                        session_id=session_id,
                        rr=rr,
                        participant_identity=participant_identity,
                        streamed=False,
                    )
                elif self._voice_config.behavior.stream_llm_tokens:
                    assert self._runtime is not None and self._state_store is not None
                    state = self._state_store.get_or_create(session_id)
                    await self._consume_runtime_stream(
                        rtc=rtc,
                        room=room,
                        out_source=out_source,
                        session_id=session_id,
                        req=req,
                        state=state,
                        participant_identity=participant_identity,
                    )
                else:
                    assert self._runtime is not None and self._state_store is not None
                    state = self._state_store.get_or_create(session_id)
                    runtime_response = await asyncio.to_thread(self._runtime.handle, req, state)
                    await self._deliver_assistant_payload_and_tts(
                        rtc=rtc,
                        room=room,
                        out_source=out_source,
                        session_id=session_id,
                        rr=runtime_response,
                        participant_identity=participant_identity,
                        streamed=False,
                    )
            except Exception:
                logger.exception("Runtime answer failed participant=%s", participant_identity)
                await self._publish_voice_state(room, state="error", detail="processing_failed")
                await self._publish_voice_state(room, state="idle")
                return

            await asyncio.sleep(0.12)
            await self._publish_voice_state(room, state="idle")
            logger.info("PTT turn complete participant=%s", participant_identity)

    async def _handle_legacy_text_packet(
        self, *, rtc, room, out_source, body: dict, index_name: str
    ) -> None:
        async with self._turn_lock:
            await self._publish_voice_state(room, state="processing")
            participant_id = str(body.get("participant_identity") or "unknown")
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
                verbose=self._voice_config.behavior.verbose_trace,
            )
            try:
                if self._chat_client is not None:
                    runtime_response = await asyncio.to_thread(self._chat_client.chat, lreq)
                else:
                    assert self._runtime is not None and self._state_store is not None
                    state = self._state_store.get_or_create(session_id)
                    runtime_response = await asyncio.to_thread(self._runtime.handle, lreq, state)
            except Exception:
                logger.exception("Legacy data packet processing failed")
                await self._publish_voice_state(room, state="error", detail="processing_failed")
                await self._publish_voice_state(room, state="idle")
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
            await asyncio.sleep(0.12)
            await self._publish_voice_state(room, state="idle")

    async def _publish_tts_audio(self, *, rtc, out_source, tts_audio: bytes, tts_encoding: str) -> None:
        try:
            if tts_encoding == "wav":
                with wave.open(io.BytesIO(tts_audio), "rb") as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    pcm = wf.readframes(wf.getnframes())
            elif tts_encoding == "pcm_s16le":
                sample_rate = 24000
                channels = 1
                pcm = tts_audio
            else:
                logger.warning("Unsupported TTS encoding for live publish: %s", tts_encoding)
                return
            bytes_per_frame = channels * 2
            samples_per_channel = max(1, len(pcm) // bytes_per_frame // 10)
            frame_bytes = samples_per_channel * bytes_per_frame
            pos = 0
            while pos + frame_bytes <= len(pcm):
                chunk = pcm[pos : pos + frame_bytes]
                pos += frame_bytes
                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=sample_rate,
                    num_channels=channels,
                    samples_per_channel=samples_per_channel,
                )
                await out_source.capture_frame(frame)
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
