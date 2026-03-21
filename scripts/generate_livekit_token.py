#!/usr/bin/env python3
"""
Mint a LiveKit access JWT for local dev (self-hosted).

Requires: pip install livekit-api
  (included in: pip install -e ".[voice]")

Defaults match docker-compose.yml / livekit-server --dev:
  API key: devkey
  API secret: secret
  Room: banking-support-room

Usage:
  python scripts/generate_livekit_token.py --identity banking-support-agent
  python scripts/generate_livekit_token.py --identity web-user-1

Env overrides: LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_ROOM
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta


def main() -> int:
    try:
        from livekit.api import AccessToken, VideoGrants
    except ImportError:
        print(
            "Missing dependency: install with  pip install livekit-api  "
            "or  pip install -e \".[voice]\"",
            file=sys.stderr,
        )
        return 1

    p = argparse.ArgumentParser(description="Generate LiveKit JWT for local dev")
    p.add_argument("--identity", required=True, help="Participant identity, e.g. banking-support-agent")
    p.add_argument(
        "--room",
        default=os.getenv("LIVEKIT_ROOM", "banking-support-room"),
        help="Room name (default: banking-support-room or LIVEKIT_ROOM)",
    )
    p.add_argument("--api-key", default=os.getenv("LIVEKIT_API_KEY", "devkey"))
    p.add_argument("--api-secret", default=os.getenv("LIVEKIT_API_SECRET", "secret"))
    p.add_argument("--ttl-hours", type=int, default=6, help="Token lifetime in hours")
    args = p.parse_args()

    token = (
        AccessToken(args.api_key, args.api_secret)
        .with_identity(args.identity)
        .with_grants(VideoGrants(room_join=True, room=args.room))
        .with_ttl(timedelta(hours=args.ttl_hours))
        .to_jwt()
    )
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
