from __future__ import annotations

import os
from datetime import timedelta


def livekit_ws_url() -> str:
    """Normalize LIVEKIT_URL for WebSocket signaling (ws/wss, no trailing slash)."""

    u = (os.getenv("LIVEKIT_URL") or "ws://127.0.0.1:7880").strip().rstrip("/")
    lower = u.lower()
    if lower.startswith("http://"):
        u = "ws://" + u[7:]
    elif lower.startswith("https://"):
        u = "wss://" + u[8:]
    return u


def livekit_env_config() -> dict[str, str]:
    """Public LiveKit settings for clients (no secrets)."""

    return {
        "livekit_url": livekit_ws_url(),
        "room": os.getenv("LIVEKIT_ROOM", "banking-support-room").strip(),
    }


def mint_participant_token(
    *,
    identity: str,
    room: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
    ttl_hours: int = 6,
) -> str:
    """
    Create a LiveKit access JWT (same grants as scripts/generate_livekit_token.py).
    """

    try:
        from livekit.api import AccessToken, VideoGrants
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "livekit-api is required for token minting. Install: pip install livekit-api "
            'or pip install -e ".[voice]"'
        ) from e

    key = api_key or os.getenv("LIVEKIT_API_KEY", "devkey")
    secret = api_secret or os.getenv("LIVEKIT_API_SECRET", "secret")
    r = (room or os.getenv("LIVEKIT_ROOM", "banking-support-room")).strip()
    ident = identity.strip()
    if not ident:
        raise ValueError("identity is required")

    return (
        AccessToken(key, secret)
        .with_identity(ident)
        .with_grants(VideoGrants(room_join=True, room=r))
        .with_ttl(timedelta(hours=ttl_hours))
        .to_jwt()
    )
