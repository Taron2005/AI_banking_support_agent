from voice_ai_banking_support_agent.voice.session_handler import (
    build_runtime_session_id,
    resolve_runtime_session_id,
    sanitize_runtime_session_id_override,
)


def test_session_id_mapping_is_deterministic() -> None:
    s1 = build_runtime_session_id(room_name="room-A", participant_identity="user_1")
    s2 = build_runtime_session_id(room_name="room-A", participant_identity="user_1")
    assert s1 == s2
    assert s1.startswith("lk::")


def test_sanitize_session_override_rejects_unsafe() -> None:
    assert sanitize_runtime_session_id_override("lk::r::u") == "lk::r::u"
    assert sanitize_runtime_session_id_override("bad id") is None
    assert sanitize_runtime_session_id_override("<script>") is None


def test_resolve_prefers_override_when_valid() -> None:
    base = build_runtime_session_id(room_name="room-A", participant_identity="user_1")
    assert (
        resolve_runtime_session_id(
            room_name="room-A",
            participant_identity="user_1",
            override="lk::room-A::user_1",
        )
        == "lk::room-A::user_1"
    )
    assert (
        resolve_runtime_session_id(room_name="room-A", participant_identity="user_1", override=None) == base
    )
    assert (
        resolve_runtime_session_id(room_name="room-A", participant_identity="user_1", override="nope nope")
        == base
    )

