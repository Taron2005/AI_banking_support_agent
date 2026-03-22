from __future__ import annotations

import re

_SESSION_OVERRIDE_RE = re.compile(r"^[a-zA-Z0-9_.:-]{1,256}$")


def build_runtime_session_id(*, room_name: str, participant_identity: str) -> str:
    """
    Deterministic mapping from LiveKit room/participant to runtime session ID.
    """

    def _clean(v: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", v.strip())

    return f"lk::{_clean(room_name)}::{_clean(participant_identity)}"


def sanitize_runtime_session_id_override(raw: str | None) -> str | None:
    """Allow client-supplied session id when it matches safe characters (aligns web /chat with voice)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or len(s) > 256:
        return None
    if not _SESSION_OVERRIDE_RE.match(s):
        return None
    return s


def resolve_runtime_session_id(
    *,
    room_name: str,
    participant_identity: str,
    override: str | None,
) -> str:
    cleaned = sanitize_runtime_session_id_override(override)
    if cleaned:
        return cleaned
    return build_runtime_session_id(room_name=room_name, participant_identity=participant_identity)

