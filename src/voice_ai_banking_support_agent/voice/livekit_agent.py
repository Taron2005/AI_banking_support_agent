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


@dataclass(frozen=True)
class LiveKitParticipantContext:
    room_name: str
    participant_identity: str


class LiveKitVoiceAgent:
    """
    Voice transport wrapper around existing runtime orchestrator.

    Runtime remains the decision-making brain; this layer only maps
    audio/text turns to runtime calls and returns TTS audio.

    This implementation is self-hosted LiveKit only.
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

    def process_turn(
        self,
        *,
        participant: LiveKitParticipantContext,
        payload: STTInput,
        index_name: str,
    ) -> VoiceTurnResult:
        session_id = build_runtime_session_id(
            room_name=participant.room_name,
            participant_identity=participant.participant_identity,
        )
        state = self._state_store.get_or_create(session_id)
        user_text = self._stt.transcribe(payload)
        if not (user_text or "").strip():
            logger.warning("STT returned empty text; orchestrator may refuse or mis-route.")
        elif "[mock-stt-unavailable]" in user_text:
            logger.warning(
                "STT is mock/non-audio mode for this payload. For real speech, set stt.provider=http_whisper "
                "and VOICE_STT_ENDPOINT, or send JSON text via LiveKit data packets."
            )
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

    async def _run_self_hosted_async(self, *, index_name: str) -> None:
        """
        Run LiveKit loop in self-hosted mode with real audio-track processing.

        Flow:
        remote mic audio track -> STT -> runtime -> TTS -> publish local audio track.
        """

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
        try:
            await room.connect(self._voice_config.livekit.url, token)
        except Exception as exc:
            logger.error(
                "LiveKit connect failed url=%s (check LIVEKIT_URL, token scope, and server).",
                self._voice_config.livekit.url,
            )
            raise RuntimeError(
                "Could not connect to LiveKit. Verify LIVEKIT_URL, LIVEKIT_TOKEN, and that the server is running."
            ) from exc

        # Publish one agent output audio track for synthesized responses.
        out_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
        out_track = rtc.LocalAudioTrack.create_audio_track("assistant-tts", out_source)
        await room.local_participant.publish_track(out_track)
        logger.info("Published assistant audio track.")

        turn_window_seconds = 3.0
        min_bytes_for_turn = 3200  # ~100ms mono int16@16k

        @room.on("track_subscribed")
        def _on_track_subscribed(track, publication, participant):  # pragma: no cover
            if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
                return
            logger.info("Subscribed remote audio track participant=%s", participant.identity)
            asyncio.create_task(
                self._consume_remote_audio_track(
                    rtc=rtc,
                    room=room,
                    track=track,
                    participant_identity=participant.identity,
                    index_name=index_name,
                    out_source=out_source,
                    turn_window_seconds=turn_window_seconds,
                    min_bytes_for_turn=min_bytes_for_turn,
                )
            )

        @room.on("data_received")
        def _on_data(packet) -> None:  # pragma: no cover
            # Keep text data mode as fallback/manual test channel.
            asyncio.create_task(self._handle_data_packet(room=room, packet=packet, index_name=index_name))

        logger.info("LiveKit agent connected; waiting for audio tracks / data packets.")
        while True:  # pragma: no cover
            await asyncio.sleep(1)

    async def _handle_data_packet(self, *, room, packet, index_name: str) -> None:
        try:
            body = json.loads(bytes(packet.data).decode("utf-8"))
            participant_id = body.get("participant_identity") or "unknown"
            text = body.get("text", "")
            result = self.process_turn(
                participant=LiveKitParticipantContext(
                    room_name=self._voice_config.livekit.room_name,
                    participant_identity=participant_id,
                ),
                payload=STTInput(content=text.encode("utf-8"), encoding="text", language=self._voice_config.stt.language),
                index_name=index_name,
            )
            response_body = json.dumps(
                {
                    "session_id": result.session_id,
                    "user_text": result.user_text,
                    "answer_text": result.runtime_response.answer_text,
                    "status": result.runtime_response.status,
                    "refusal_reason": result.runtime_response.refusal_reason,
                    "decision_trace": result.runtime_response.decision_trace,
                    "tts_audio_bytes": len(result.tts_output.audio),
                },
                ensure_ascii=False,
            ).encode("utf-8")
            await room.local_participant.publish_data(response_body, reliable=True, topic="assistant.text")
        except Exception:
            logger.exception("Failed to process livekit data packet")

    async def _consume_remote_audio_track(
        self,
        *,
        rtc,
        room,
        track,
        participant_identity: str,
        index_name: str,
        out_source,
        turn_window_seconds: float,
        min_bytes_for_turn: int,
    ) -> None:
        pcm_chunks: list[bytes] = []
        chunk_duration = 0.0
        stream = rtc.AudioStream(track=track, sample_rate=16000, num_channels=1)
        async for evt in stream:  # pragma: no cover
            frame = evt.frame
            pcm = bytes(frame.data)
            pcm_chunks.append(pcm)
            chunk_duration += frame.duration
            if chunk_duration < turn_window_seconds:
                continue
            pcm_blob = b"".join(pcm_chunks)
            pcm_chunks = []
            chunk_duration = 0.0
            if len(pcm_blob) < min_bytes_for_turn:
                continue
            wav_bytes = self._pcm_to_wav(
                pcm_blob,
                sample_rate=16000,
                channels=1,
            )
            result = self.process_turn(
                participant=LiveKitParticipantContext(
                    room_name=self._voice_config.livekit.room_name,
                    participant_identity=participant_identity,
                ),
                payload=STTInput(
                    content=wav_bytes,
                    encoding="wav",
                    language=self._voice_config.stt.language,
                ),
                index_name=index_name,
            )
            # Publish text metadata packet for debug UI clients.
            packet = json.dumps(
                {
                    "session_id": result.session_id,
                    "user_text": result.user_text,
                    "answer_text": result.runtime_response.answer_text,
                    "status": result.runtime_response.status,
                    "refusal_reason": result.runtime_response.refusal_reason,
                    "decision_trace": result.runtime_response.decision_trace,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            await room.local_participant.publish_data(packet, reliable=True, topic="assistant.text")
            await self._publish_tts_audio(rtc=rtc, out_source=out_source, tts_audio=result.tts_output.audio, tts_encoding=result.tts_output.encoding)

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
            samples_per_channel = max(1, len(pcm) // bytes_per_frame // 10)  # ~100ms frames
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

