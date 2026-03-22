"""Lifecycle helpers for multi-turn mic consumption (no LiveKit runtime required)."""

import asyncio

import pytest

from voice_ai_banking_support_agent.voice.livekit_mic import cancel_audio_consumer_task, find_remote_audio_track


class _FakePub:
    def __init__(self, kind, track) -> None:
        self.kind = kind
        self.track = track


class _FakeParticipant:
    def __init__(self, pubs: dict) -> None:
        self.track_publications = pubs


class _FakeRoom:
    def __init__(self, participants: dict) -> None:
        self.remote_participants = participants


def test_find_remote_audio_track_requires_livekit_enum() -> None:
    try:
        from livekit import rtc
    except ImportError:
        pytest.skip("livekit not installed")
    audio = object()
    pub_ok = _FakePub(rtc.TrackKind.KIND_AUDIO, audio)
    pub_vid = _FakePub(rtc.TrackKind.KIND_VIDEO, object())
    room = _FakeRoom({"u1": _FakeParticipant({"a": pub_ok, "v": pub_vid})})
    assert find_remote_audio_track(room, participant_identity="u1") is audio
    assert find_remote_audio_track(room, participant_identity="missing") is None


def test_cancel_audio_consumer_task_done() -> None:
    async def _run() -> None:
        async def quick() -> None:
            await asyncio.sleep(0)

        t = asyncio.create_task(quick())
        await t
        await cancel_audio_consumer_task(t)

    asyncio.run(_run())


def test_cancel_audio_consumer_task_cancelled() -> None:
    async def _run() -> None:
        async def slow() -> None:
            await asyncio.sleep(10)

        t = asyncio.create_task(slow())
        await cancel_audio_consumer_task(t)

    asyncio.run(_run())
