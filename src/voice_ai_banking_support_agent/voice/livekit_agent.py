from __future__ import annotations

import asyncio
import io
import json
import logging
import wave
from dataclasses import dataclass

from ..runtime.orchestrator import RuntimeOrchestrator, RuntimeRequest
from ..runtime.session_state import SessionStateStore
from .session_handler import build_runtime_session_id
from .stt import STTProvider
from .tts import TTSProvider
from .voice_config import VoiceConfig
from .voice_models import STTInput, VoiceTurnResult

logger = logging.getLogger(__name__)

TOPIC_PTT = "voice.ptt"
TOPIC_ASSISTANT_TEXT = "assistant.text"
TOPIC_VOICE_STATE = "voice.state"
TOPIC_VOICE_TRANSCRIPT_FINAL = "voice.transcript.final"


@dataclass(frozen=True)
class LiveKitParticipantContext:
    room_name: str
    participant_identity: str


class LiveKitVoiceAgent:
    """
    Voice transport wrapper around existing runtime orchestrator.

    Push-to-talk: the browser sends `voice.ptt` data packets (`start` / `end`) and only
    publishes mic audio while recording. The agent buffers PCM while `start` is active
    and runs STT → runtime → TTS once on `end`.
    """

    def __init__(
        self,
        *,
        runtime: RuntimeOrchestrator,
        state_store: SessionStateStore,
        stt_provider: STTProvider,
        tts_provider: TTSProvider,
        voice_config: VoiceConfig,
    ) -> None:
        self._runtime = runtime
        self._state_store = state_store
        self._stt = stt_provider
        self._tts = tts_provider
        self._voice_config = voice_config
        self._turn_lock = asyncio.Lock()
        self._ptt_recording: dict[str, bool] = {}
        self._ptt_buffers: dict[str, list[bytes]] = {}
        self._audio_consumer_tasks: dict[str, asyncio.Task[None]] = {}

    def process_turn(
        self,
        *,
        participant: LiveKitParticipantContext,
        payload: STTInput,
        index_name: str,
    ) -> VoiceTurnResult:
        if payload.encoding == "text":
            user_text = payload.content.decode("utf-8", errors="ignore").strip()
        else:
            user_text = self._stt.transcribe(payload)
            if not (user_text or "").strip():
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
        token = self._load_livekit_token()
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
            self._ptt_recording[pid] = True
            self._ptt_buffers[pid] = []
            logger.info("PTT start participant=%s", pid)
            await self._publish_voice_state(room, state="listening")
            return

        if ptype == "end":
            asyncio.create_task(
                self._finalize_ptt_turn(
                    rtc=rtc,
                    room=room,
                    out_source=out_source,
                    participant_identity=pid,
                    index_name=index_name,
                )
            )
            return

        logger.warning("Unknown PTT type=%s", ptype)

    async def _finalize_ptt_turn(
        self,
        *,
        rtc,
        room,
        out_source,
        participant_identity: str,
        index_name: str,
    ) -> None:
        async with self._turn_lock:
            self._ptt_recording[participant_identity] = False
            await asyncio.sleep(0.2)
            chunks = self._ptt_buffers.pop(participant_identity, [])
            pcm_blob = b"".join(chunks)
            await self._publish_voice_state(room, state="processing")

            min_bytes = 3200  # ~100ms mono int16@16k
            if len(pcm_blob) < min_bytes:
                logger.info("PTT end: audio too short participant=%s bytes=%s", participant_identity, len(pcm_blob))
                await self._publish_voice_state(room, state="error", detail="no_speech_detected")
                await self._publish_voice_state(room, state="idle")
                return

            wav_bytes = self._pcm_to_wav(pcm_blob, sample_rate=16000, channels=1)
            participant = LiveKitParticipantContext(
                room_name=self._voice_config.livekit.room_name,
                participant_identity=participant_identity,
            )
            await self._publish_voice_state(room, state="processing", detail="transcribing")
            try:
                user_text = self._stt.transcribe(
                    STTInput(
                        content=wav_bytes,
                        encoding="wav",
                        language=self._voice_config.stt.language,
                    )
                )
            except Exception:
                logger.exception("STT failed participant=%s", participant_identity)
                await self._publish_voice_state(room, state="error", detail="stt_failed")
                await self._publish_voice_state(room, state="idle")
                return

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
            try:
                result = self._run_runtime_and_synthesize(
                    participant=participant,
                    user_text=user_text,
                    index_name=index_name,
                )
            except Exception:
                logger.exception("runtime/TTS failed participant=%s", participant_identity)
                await self._publish_voice_state(room, state="error", detail="processing_failed")
                await self._publish_voice_state(room, state="idle")
                return

            await self._publish_voice_state(room, state="speaking")
            packet = json.dumps(
                {
                    "session_id": result.session_id,
                    "answer_text": result.runtime_response.answer_text,
                    "status": result.runtime_response.status,
                    "refusal_reason": result.runtime_response.refusal_reason,
                    "decision_trace": result.runtime_response.decision_trace,
                    "tts_audio_bytes": len(result.tts_output.audio),
                },
                ensure_ascii=False,
            ).encode("utf-8")
            await room.local_participant.publish_data(packet, reliable=True, topic=TOPIC_ASSISTANT_TEXT)

            await self._publish_tts_audio(
                rtc=rtc, out_source=out_source, tts_audio=result.tts_output.audio, tts_encoding=result.tts_output.encoding
            )

            await self._publish_voice_state(room, state="idle")
            logger.info("PTT turn complete participant=%s", participant_identity)

    async def _handle_legacy_text_packet(
        self, *, rtc, room, out_source, body: dict, index_name: str
    ) -> None:
        async with self._turn_lock:
            await self._publish_voice_state(room, state="processing")
            participant_id = str(body.get("participant_identity") or "unknown")
            text = str(body.get("text", "") or "")
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
            try:
                result = self._run_runtime_and_synthesize(
                    participant=participant,
                    user_text=text.strip(),
                    index_name=index_name,
                )
            except Exception:
                logger.exception("Legacy data packet processing failed")
                await self._publish_voice_state(room, state="error", detail="processing_failed")
                await self._publish_voice_state(room, state="idle")
                return

            await self._publish_voice_state(room, state="speaking")
            response_body = json.dumps(
                {
                    "session_id": result.session_id,
                    "answer_text": result.runtime_response.answer_text,
                    "status": result.runtime_response.status,
                    "refusal_reason": result.runtime_response.refusal_reason,
                    "decision_trace": result.runtime_response.decision_trace,
                    "tts_audio_bytes": len(result.tts_output.audio),
                },
                ensure_ascii=False,
            ).encode("utf-8")
            await room.local_participant.publish_data(response_body, reliable=True, topic=TOPIC_ASSISTANT_TEXT)
            await self._publish_tts_audio(
                rtc=rtc, out_source=out_source, tts_audio=result.tts_output.audio, tts_encoding=result.tts_output.encoding
            )
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

    @staticmethod
    def _load_livekit_token() -> str:
        import os

        token = os.getenv("LIVEKIT_TOKEN")
        if not token:
            raise RuntimeError("LIVEKIT_TOKEN is required for self-hosted room connection.")
        return token
