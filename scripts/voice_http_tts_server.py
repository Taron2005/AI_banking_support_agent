"""
Local Armenian TTS HTTP server matching voice/tts.py HTTPTTSProvider.

Contract:
  POST /synthesize
    JSON: {"text": str, "language": str, "voice": str, "format": "wav"}
    response: application/json {"audio_base64": "<wav bytes base64>"}

Uses Microsoft Edge TTS (edge-tts) + miniaudio to decode MP3 to PCM and wrap WAV
(no ffmpeg — pure Python wheels on common platforms).

Armenian note: ``hy-AM-*`` neural voices are often *missing* from Edge's voice list for some
regions/networks, which yields ``NoAudioReceived``. This server automatically retries with
multilingual English voices that still accept Armenian script (sounds accented but works).

Run:
  pip install -e ".[voice_local_servers]"
  python scripts/voice_http_tts_server.py

Optional env:
  EDGE_TTS_FALLBACK_VOICES=en-US-AndrewMultilingualNeural,en-US-GuyNeural

Then in .env:
  VOICE_TTS_ENDPOINT=http://127.0.0.1:8089/synthesize
  VOICE_TTS_VOICE=hy-AM-AnahitNeural   # optional; or pass voice in JSON from voice_config.yaml
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import wave

# Must be module-level: nested `synthesize()` annotations are resolved against *this* module's
# globals. If `Request` were imported only inside main(), FastAPI treats the param as a query
# string → 422 "Field required" for http_request (PEP 563 postponed annotations).
from starlette.requests import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_VOICE = "hy-AM-AnahitNeural"
# When hy-AM is not offered by Microsoft for this IP/region, these still return audio for UTF-8 text.
DEFAULT_FALLBACK_VOICES = (
    "en-US-AndrewMultilingualNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-GuyNeural",
)
TTS_SAMPLE_RATE = 24000


def _parse_fallback_voices(arg: str | None) -> list[str]:
    raw = (arg or "").strip()
    if not raw:
        raw = os.environ.get("EDGE_TTS_FALLBACK_VOICES", "")
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        v = part.strip()
        if v and v not in out:
            out.append(v)
    if not out:
        out = list(DEFAULT_FALLBACK_VOICES)
    return out


async def _edge_stream_to_mp3(edge_tts_mod: object, text: str, voice: str) -> bytes:
    communicate = edge_tts_mod.Communicate(text, voice)
    mp3_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]
    return mp3_data


async def _synthesize_mp3_with_fallback(
    edge_tts_mod: object,
    text: str,
    primary_voice: str,
    fallbacks: list[str],
) -> tuple[bytes, str, list[str]]:
    """Return (mp3_bytes, voice_used, errors_per_attempt)."""
    errors: list[str] = []
    no_audio_exc = edge_tts_mod.exceptions.NoAudioReceived

    chain: list[str] = []
    for v in [primary_voice] + fallbacks:
        if v and v not in chain:
            chain.append(v)

    for voice in chain:
        try:
            mp3 = await _edge_stream_to_mp3(edge_tts_mod, text, voice)
            if mp3:
                if voice != primary_voice:
                    logger.warning(
                        "edge-tts: primary voice %r returned no audio; used fallback %r",
                        primary_voice,
                        voice,
                    )
                return mp3, voice, errors
            errors.append(f"{voice}: empty audio stream")
        except no_audio_exc:
            errors.append(f"{voice}: NoAudioReceived")
            logger.info("edge-tts attempt failed voice=%s: %s", voice, errors[-1])
        except Exception as exc:  # noqa: BLE001 — surface all edge-tts / network failures
            errors.append(f"{voice}: {type(exc).__name__}: {exc}")
            logger.info("edge-tts attempt failed voice=%s: %s", voice, errors[-1])

    return b"", "", errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8089)
    p.add_argument("--voice", default=DEFAULT_VOICE, help="edge-tts voice id (preferred)")
    p.add_argument(
        "--fallback-voices",
        default="",
        help="Comma-separated Edge voices to try if primary fails (default: built-in multilingual list).",
    )
    args = p.parse_args()
    fallback_list = _parse_fallback_voices(args.fallback_voices)

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

    from fastapi import FastAPI
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

    # strict_content_type=False: accept JSON bodies when clients send application/json; charset=utf-8
    app = FastAPI(title="Local TTS", version="0.1", strict_content_type=False)

    @app.post("/synthesize")
    async def synthesize(http_request: Request):
        # Use http_request (not "request"): some FastAPI builds mis-resolve bare `request` as a query param.
        # Dict body + loose parsing avoids 422 when clients send nulls, alternate keys, or
        # Pydantic/OpenAPI quirks around a JSON field named "format".
        try:
            body = await http_request.json()
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

        mp3_data, used_voice, attempt_errors = await _synthesize_mp3_with_fallback(
            edge_tts,
            text,
            voice,
            fallback_list,
        )
        if not mp3_data:
            return JSONResponse(
                {
                    "error": "edge_tts_all_voices_failed",
                    "detail": (
                        "Microsoft Edge TTS did not return audio for any configured voice. "
                        "Armenian (hy-AM) voices are often unavailable by region; fallbacks should "
                        "still work — check firewall / proxy, or set EDGE_TTS_FALLBACK_VOICES."
                    ),
                    "attempts": attempt_errors,
                },
                status_code=502,
            )

        try:
            wav_bytes = mp3_bytes_to_wav(mp3_data)
        except miniaudio.DecodeError as exc:
            logger.exception("miniaudio failed to decode edge-tts MP3")
            return JSONResponse(
                {"error": "mp3_decode_failed", "detail": str(exc)},
                status_code=502,
            )

        b64 = base64.b64encode(wav_bytes).decode("ascii")
        payload: dict = {"audio_base64": b64}
        if used_voice != voice:
            payload["voice_used"] = used_voice
            payload["voice_requested"] = voice
        return payload

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": "voice-http-tts",
            "default_voice": args.voice,
            "fallback_voices": fallback_list,
            "handler": "starlette-request-json-body-v2",
        }

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
