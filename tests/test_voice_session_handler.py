from voice_ai_banking_support_agent.voice.session_handler import build_runtime_session_id


def test_session_id_mapping_is_deterministic() -> None:
    s1 = build_runtime_session_id(room_name="room-A", participant_identity="user_1")
    s2 = build_runtime_session_id(room_name="room-A", participant_identity="user_1")
    assert s1 == s2
    assert s1.startswith("lk::")

