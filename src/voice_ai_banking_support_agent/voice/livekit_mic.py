"""
Remote microphone track lookup and audio consumer lifecycle helpers.

Root cause addressed: ``AudioStream`` iteration ends when the publisher unpublishes or the
SDK closes the stream; a one-shot consumer task then exits. The next PTT turn must start
a **fresh** consumer on the current ``RemoteAudioTrack`` (see LiveKitVoiceAgent).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def find_remote_audio_track(room: Any, *, participant_identity: str) -> Any | None:
    """Return the first subscribed remote audio track for this participant, or None."""

    try:
        from livekit import rtc
    except ImportError:
        return None

    try:
        rps = room.remote_participants
    except Exception:
        return None
    rp = rps.get(participant_identity)
    if rp is None:
        return None
    try:
        pubs = rp.track_publications
    except Exception:
        return None
    for pub in pubs.values():
        try:
            if pub.kind != rtc.TrackKind.KIND_AUDIO:
                continue
        except Exception:
            continue
        track = getattr(pub, "track", None)
        if track is not None:
            return track
    return None


async def cancel_audio_consumer_task(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    if task.done():
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Prior audio consumer ended with error", exc_info=True)
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("Audio consumer cancel await", exc_info=True)


async def wait_for_remote_audio_track(
    room: Any,
    *,
    participant_identity: str,
    max_wait_s: float,
    poll_interval_s: float = 0.05,
) -> Any | None:
    """Poll until a subscribed audio track exists (handles publish-after-PTT-start race)."""

    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        t = find_remote_audio_track(room, participant_identity=participant_identity)
        if t is not None:
            return t
        await asyncio.sleep(poll_interval_s)
    return None
