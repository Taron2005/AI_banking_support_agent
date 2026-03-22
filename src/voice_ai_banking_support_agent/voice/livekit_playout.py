"""
Decode TTS payloads and push PCM into LiveKit ``AudioSource`` at a single fixed sample rate.

The agent publishes one ``LocalAudioTrack``; ``AudioSource`` is created at
``livekit_publish_sample_rate`` (default 24 kHz). TTS backends may return WAV at 22.05/44.1/48 kHz
or stereo — we normalize to mono s16le at the publish rate so playout is not chipmunk/slow or
silent due to format mismatch.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _linear_resample_mono_f32(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr <= 0 or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    in_idx = np.arange(len(audio), dtype=np.float64)
    out_len = max(1, int(round(len(audio) * float(dst_sr) / float(src_sr))))
    out_idx = np.linspace(0.0, len(audio) - 1.0, num=out_len, dtype=np.float64)
    return np.interp(out_idx, in_idx, audio.astype(np.float64)).astype(np.float32)


def _float_mono_to_s16le(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _wav_bytes_to_mono_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sw = wf.getsampwidth()
        if sw != 2:
            raise ValueError(f"expected 16-bit PCM WAV, got sampwidth={sw}")
        ch = wf.getnchannels()
        rate = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, rate


def _decode_mp3_to_mono_float32(mp3_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        import miniaudio
    except ImportError as exc:
        raise RuntimeError(
            "TTS returned MP3 but miniaudio is not installed. "
            'Use WAV from your TTS server (format: wav) or pip install miniaudio — e.g. pip install -e ".[voice_local_servers]"'
        ) from exc
    decoded = miniaudio.decode(mp3_bytes)
    dr = int(decoded.sample_rate)
    ch = max(1, int(decoded.nchannels))
    if decoded.sample_format != miniaudio.SampleFormat.SIGNED16:
        raise RuntimeError(
            f"miniaudio decode returned sample_format={decoded.sample_format}; expected SIGNED16"
        )
    raw = decoded.samples.tobytes()
    if not raw:
        return np.array([], dtype=np.float32), dr
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, dr


def tts_bytes_to_mono_s16le_at_rate(
    audio: bytes,
    *,
    encoding: str,
    target_sample_rate: int,
    pcm_assumed_rate_if_raw: int = 24000,
) -> bytes:
    """
    Return mono PCM s16le at ``target_sample_rate`` ready for ``AudioSource.capture_frame``.
    """

    enc = (encoding or "wav").strip().lower()
    if enc == "pcm_s16le":
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        sr = int(pcm_assumed_rate_if_raw)
        x = _linear_resample_mono_f32(x, sr, target_sample_rate)
        return _float_mono_to_s16le(x)

    if enc == "wav":
        x, sr = _wav_bytes_to_mono_float32(audio)
        x = _linear_resample_mono_f32(x, sr, target_sample_rate)
        return _float_mono_to_s16le(x)

    if enc == "mp3":
        x, sr = _decode_mp3_to_mono_float32(audio)
        x = _linear_resample_mono_f32(x, sr, target_sample_rate)
        return _float_mono_to_s16le(x)

    raise ValueError(f"unsupported TTS encoding for LiveKit playout: {encoding!r}")


async def publish_pcm_s16le_to_audio_source(
    rtc: Any,
    out_source: Any,
    pcm_s16le_mono: bytes,
    *,
    sample_rate: int,
    num_channels: int = 1,
    frame_ms: float = 20.0,
    pace_realtime: bool = True,
) -> None:
    """
    Push mono s16le PCM to ``AudioSource`` in fixed-size frames.

    ``pace_realtime`` sleeps ~frame_ms between frames so the remote client hears continuous audio
    instead of a single burst (reduces underruns/glitches on some networks).
    """

    if num_channels != 1:
        raise ValueError("LiveKit assistant track is mono; use mono PCM.")
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("sample_rate must be positive")
    bytes_per_frame_unit = 2 * num_channels
    samples_per_tick = max(1, int(round(sr * float(frame_ms) / 1000.0)))
    chunk_bytes = samples_per_tick * bytes_per_frame_unit
    pcm = pcm_s16le_mono
    pos = 0

    while pos < len(pcm):
        end = min(pos + chunk_bytes, len(pcm))
        piece = pcm[pos:end]
        pos = end
        if len(piece) < chunk_bytes:
            piece = piece + b"\x00" * (chunk_bytes - len(piece))
        samples_this = chunk_bytes // bytes_per_frame_unit
        frame = rtc.AudioFrame(
            data=piece,
            sample_rate=sr,
            num_channels=num_channels,
            samples_per_channel=samples_this,
        )
        await out_source.capture_frame(frame)
        if pace_realtime and pos < len(pcm):
            await asyncio.sleep(float(frame_ms) / 1000.0)
