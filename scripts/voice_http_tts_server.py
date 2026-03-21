"""
Local Armenian TTS HTTP server matching voice/tts.py HTTPTTSProvider.

Contract:
  POST /synthesize
    JSON: {"text": str, "language": str, "voice": str, "format": "wav"}
    response: application/json {"audio_base64": "<wav bytes base64>"}

Uses Microsoft Edge TTS (edge-tts) + miniaudio to decode MP3 to PCM and wrap WAV
(no ffmpeg — pure Python wheels on common platforms).

Run:
  pip install -e ".[voice_local_servers]"
  python scripts/voice_http_tts_server.py

Then in .env:
  VOICE_TTS_ENDPOINT=http://127.0.0.1:8089/synthesize
  VOICE_TTS_VOICE=hy-AM-AnahitNeural   # optional; or pass voice in JSON from voice_config.yaml
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import wave

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_VOICE = "hy-AM-AnahitNeural"
TTS_SAMPLE_RATE = 24000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8089)
    p.add_argument("--voice", default=DEFAULT_VOICE, help="edge-tts voice id")
    args = p.parse_args()

    try:
        import edge_tts
    except ImportError as e:
        raise SystemExit("Install: pip install edge-tts") from e

    try:
        import miniaudio
    except ImportError as e:
        raise SystemExit(
            "Install: pip install miniaudio (or pip install -e \".[voice_local_servers]\")"
        ) from e

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    def mp3_bytes_to_wav(mp3_data: bytes) -> bytes:
        decoded = miniaudio.decode(
            mp3_data,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=TTS_SAMPLE_RATE,
        )
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(decoded.nchannels)
            wf.setsampwidth(2)
            wf.setframerate(decoded.sample_rate)
            wf.writeframes(decoded.samples.tobytes())
        return buf.getvalue()

    app = FastAPI(title="Local TTS", version="0.1")

    @app.post("/synthesize")
    async def synthesize(request: Request):
        # Dict body + loose parsing avoids 422 when clients send nulls, alternate keys, or
        # Pydantic/OpenAPI quirks around a JSON field named "format".
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_json"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body_must_be_object"}, status_code=400)

        text = str(body.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "empty text"}, status_code=400)

        _ = str(body.get("language") or "hy-AM").strip()  # reserved for future non-Edge backends
        voice_raw = body.get("voice")
        if voice_raw is None or (isinstance(voice_raw, str) and not voice_raw.strip()):
            voice_raw = body.get("voice_name")
        voice = str(voice_raw or "").strip()
        voice = (voice or args.voice or DEFAULT_VOICE).strip()
        if voice.lower() == "default":
            voice = DEFAULT_VOICE

        communicate = edge_tts.Communicate(text, voice)
        mp3_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data += chunk["data"]
        if not mp3_data:
            return JSONResponse({"error": "no audio from edge-tts"}, status_code=502)

        try:
            wav_bytes = mp3_bytes_to_wav(mp3_data)
        except miniaudio.DecodeError as exc:
            logger.exception("miniaudio failed to decode edge-tts MP3")
            return JSONResponse(
                {"error": "mp3_decode_failed", "detail": str(exc)},
                status_code=502,
            )

        b64 = base64.b64encode(wav_bytes).decode("ascii")
        return {"audio_base64": b64}

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "voice-http-tts", "default_voice": args.voice}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
