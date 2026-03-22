import io
import wave

import numpy as np

from voice_ai_banking_support_agent.voice.livekit_playout import (
    tts_bytes_to_mono_s16le_at_rate,
)


def _sine_wav_mono_s16(sr: int, n: int, freq: float = 440.0) -> bytes:
    t = np.arange(n, dtype=np.float64) / float(sr)
    y = (np.sin(2 * np.pi * freq * t) * 0.2 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y.tobytes())
    return buf.getvalue()


def test_tts_wav_resample_48k_to_24k_length_order() -> None:
    wav = _sine_wav_mono_s16(48000, 4800)
    out = tts_bytes_to_mono_s16le_at_rate(wav, encoding="wav", target_sample_rate=24000)
    assert len(out) == 2400 * 2


def test_tts_pcm_s16le_assumed_rate_resample() -> None:
    x = np.zeros(2400, dtype=np.int16)
    x[::100] = 10000
    raw = x.tobytes()
    out = tts_bytes_to_mono_s16le_at_rate(
        raw, encoding="pcm_s16le", target_sample_rate=24000, pcm_assumed_rate_if_raw=48000
    )
    assert len(out) == 1200 * 2
