from voice_ai_banking_support_agent.runtime.models import RuntimeResponse
from voice_ai_banking_support_agent.runtime.session_state import SessionStateStore
from voice_ai_banking_support_agent.voice.livekit_agent import LiveKitParticipantContext, LiveKitVoiceAgent
from voice_ai_banking_support_agent.voice.stt import MockSTTProvider
from voice_ai_banking_support_agent.voice.tts import MockTTSProvider
from voice_ai_banking_support_agent.voice.voice_config import VoiceConfig
from voice_ai_banking_support_agent.voice.voice_models import STTInput


class FakeRuntime:
    def __init__(self, status: str) -> None:
        self._status = status

    def handle(self, req, state):  # noqa: ANN001
        if self._status == "refused":
            return RuntimeResponse(
                answer_text="Կներեք, չի աջակցվում։",
                status="refused",
                refusal_reason="out_of_scope",
                detected_topic="out_of_scope",
            )
        return RuntimeResponse(
            answer_text="Սա պատասխանն է։",
            status="answered",
            detected_topic="deposit",
        )


def _agent(status: str) -> LiveKitVoiceAgent:
    return LiveKitVoiceAgent(
        runtime=FakeRuntime(status),  # type: ignore[arg-type]
        state_store=SessionStateStore(),
        stt_provider=MockSTTProvider(),
        tts_provider=MockTTSProvider(),
        voice_config=VoiceConfig(),
    )


def test_voice_agent_propagates_answer_status() -> None:
    agent = _agent("answered")
    out = agent.process_turn(
        participant=LiveKitParticipantContext(room_name="r", participant_identity="u1"),
        payload=STTInput(content="Ի՞նչ ավանդներ կան".encode("utf-8"), encoding="text"),
        index_name="idx",
    )
    assert out.runtime_response.status == "answered"
    assert len(out.tts_output.audio) > 0


def test_voice_agent_propagates_refusal_status() -> None:
    agent = _agent("refused")
    out = agent.process_turn(
        participant=LiveKitParticipantContext(room_name="r", participant_identity="u1"),
        payload=STTInput(content="best bank?".encode("utf-8"), encoding="text"),
        index_name="idx",
    )
    assert out.runtime_response.status == "refused"
    assert "Կներեք" in out.runtime_response.answer_text

