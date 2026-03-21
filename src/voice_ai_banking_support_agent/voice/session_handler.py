from __future__ import annotations

import re


def build_runtime_session_id(*, room_name: str, participant_identity: str) -> str:
    """
    Deterministic mapping from LiveKit room/participant to runtime session ID.
    """

    def _clean(v: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", v.strip())

    return f"lk::{_clean(room_name)}::{_clean(participant_identity)}"

