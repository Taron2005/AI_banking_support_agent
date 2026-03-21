"""
Local Armenian-capable STT HTTP server matching voice/stt.py HTTPWhisperSTTProvider.

Contract:
  POST /transcribe
    multipart: field "file" (WAV/PCM), form "language" (default hy)
    response: {"text": "..."}  UTF-8 JSON

Run (from repo root, after deps):
  pip install faster-whisper python-multipart
  python scripts/voice_http_stt_server.py

Then in .env:
  VOICE_STT_ENDPOINT=http://127.0.0.1:8088/transcribe
"""
from __future__ import annotations

import argparse
import io
import logging
import wave

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _wav_bytes_to_float32_mono_16k(raw: bytes) -> np.ndarray:
    """Decode WAV (s16le) to mono float32 at 16 kHz — Whisper/faster-whisper expectation for ndarray input."""
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sw = wf.getsampwidth()
        if sw != 2:
            raise ValueError(f"expected 16-bit PCM WAV, sampwidth={sw}")
        ch = wf.getnchannels()
        rate = wf.getframerate()
        n = wf.getnframes()
        pcm = wf.readframes(n)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if rate <= 0:
        raise ValueError(f"invalid sample rate {rate}")
    if rate != 16000 and audio.size > 0:
        # Linear resample to 16 kHz (agent publishes 16 kHz; tolerate minor drift / odd files)
        in_idx = np.arange(len(audio), dtype=np.float64)
        out_len = max(1, int(round(len(audio) * 16000.0 / rate)))
        out_idx = np.linspace(0.0, len(audio) - 1.0, num=out_len, dtype=np.float64)
        audio = np.interp(out_idx, in_idx, audio.astype(np.float64)).astype(np.float32)
    return audio


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--model", default="small", help="faster-whisper model: tiny, base, small, medium, large-v3, ...")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--compute-type", default="int8", dest="compute_type")
    args = p.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit(
            "Install: pip install faster-whisper\n"
            "Optional GPU: pip install faster-whisper and use --device cuda --compute-type float16"
        ) from e

    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse
    import uvicorn

    logger.info("Loading Whisper model=%s device=%s …", args.model, args.device)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    app = FastAPI(title="Local STT", version="0.1")

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        language: str = Form("hy"),
    ):
        raw = await file.read()
        if not raw:
            return {"text": ""}
        lang = (language or "hy").strip().lower()
        if lang.startswith("hy"):
            lang = "hy"
        try:
            audio = _wav_bytes_to_float32_mono_16k(raw)
        except Exception as exc:
            logger.exception("WAV decode failed (need s16le WAV from the voice agent)")
            return JSONResponse(
                {"error": "invalid_wav", "detail": str(exc)},
                status_code=400,
            )
        if audio.size == 0:
            return {"text": ""}
        try:
            segments, _info = model.transcribe(
                audio,
                language=lang if lang else None,
                vad_filter=False,
            )
            text = "".join(s.text for s in segments).strip()
            return {"text": text}
        except Exception as exc:
            logger.exception("Whisper transcribe failed")
            return JSONResponse(
                {"error": "transcribe_failed", "detail": str(exc)},
                status_code=500,
            )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "voice-http-stt"}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
