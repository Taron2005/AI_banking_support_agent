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

Default Whisper checkpoint is ``small`` (env ``VOICE_WHISPER_MODEL``); use ``base`` or ``tiny`` for fastest load, ``medium`` for stronger Armenian.
Decode latency: env ``VOICE_WHISPER_BEAM`` (default ``1``), optional ``VOICE_WHISPER_BEAM_RETRY`` (default ``3``) for a second pass if the first decode is empty.

VAD uses onnxruntime (Silero). If Windows reports DLL load errors for onnxruntime, use
  python scripts/voice_http_stt_server.py --vad-filter off
or reinstall a cp310 wheel and keep protobuf<6 for Gemini/grpc:
  pip install "protobuf>=5.26.1,<6"
  pip install --no-deps --force-reinstall "onnxruntime==1.23.2"
"""
import argparse
import asyncio
import io
import logging
import os
import threading
import wave

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

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


def _onnxruntime_usable() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument(
        "--model",
        default=(os.environ.get("VOICE_WHISPER_MODEL") or "small").strip(),
        help="faster-whisper model (env VOICE_WHISPER_MODEL overrides). "
        "small balances speed and quality; base/tiny load fastest; medium/large-v3 are stronger for Armenian.",
    )
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--compute-type", default="int8", dest="compute_type")
    p.add_argument(
        "--vad-filter",
        choices=("auto", "on", "off"),
        default="auto",
        help="Silero VAD inside faster-whisper: needs onnxruntime. 'auto' enables VAD only if onnxruntime loads.",
    )
    p.add_argument(
        "--beam-size",
        type=int,
        default=0,
        help="faster-whisper beam_size for the first decode pass (0 = env VOICE_WHISPER_BEAM or 1). "
        "Use 1 for lowest latency; 3–5 for harder Armenian at more CPU cost.",
    )
    args = p.parse_args()

    if args.vad_filter == "on":
        vad_filter = True
    elif args.vad_filter == "off":
        vad_filter = False
    else:
        vad_filter = _onnxruntime_usable()
        if not vad_filter:
            logger.warning(
                "onnxruntime not usable; transcribe will use vad_filter=False. "
                "Fix ONNX (see docstring) or pass --vad-filter on after repairing it."
            )

    def _beam_from_env(primary: int) -> int:
        if primary > 0:
            return max(1, min(primary, 8))
        raw = (os.environ.get("VOICE_WHISPER_BEAM") or "1").strip()
        try:
            return max(1, min(int(raw or "1"), 8))
        except ValueError:
            return 1

    def _beam_retry() -> int:
        raw = (os.environ.get("VOICE_WHISPER_BEAM_RETRY") or "3").strip()
        try:
            return max(1, min(int(raw or "3"), 8))
        except ValueError:
            return 3

    beam_primary = _beam_from_env(args.beam_size)
    beam_retry = _beam_retry()
    logger.info("STT decode: beam_primary=%s beam_retry=%s", beam_primary, beam_retry)

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit(
            "faster-whisper (or a native dependency) failed to import.\n"
            f"  Caused by: {e!r}\n"
            "Typical fixes:\n"
            "  pip install -e \".[voice_local_servers]\"\n"
            "If you use Python 3.10 but see cp312 (or wrong ABI) in .venv errors, reinstall native wheels, e.g.:\n"
            "  pip install --force-reinstall --no-cache-dir ctranslate2 onnxruntime av\n"
            "On Windows, PyAV DLL errors often clear with: pip install \"av>=11,<14\"\n"
            "If onnxruntime DLL fails only when transcribing, run with: --vad-filter off\n"
            "GPU: --device cuda --compute-type float16 (needs CUDA-enabled ctranslate2).\n"
        ) from e

    import uvicorn

    # Load Whisper in a background thread so uvicorn binds immediately. Otherwise /health cannot
    # respond until weights download + load finish (often several minutes; start_stack would see
    # connection refused and look like a failure).
    model_holder: dict = {"model": None, "error": None}

    def _load_whisper() -> None:
        try:
            logger.info(
                "Loading Whisper model=%s device=%s vad_filter=%s …",
                args.model,
                args.device,
                vad_filter,
            )
            model_holder["model"] = WhisperModel(
                args.model, device=args.device, compute_type=args.compute_type
            )
            logger.info("Whisper model ready.")
        except Exception:
            logger.exception("Whisper model load failed")
            model_holder["error"] = "model_load_failed"

    threading.Thread(target=_load_whisper, name="whisper-load", daemon=True).start()

    # Blocking Whisper in the asyncio event loop starves other requests (2nd PTT turn timeouts / empty STT).
    _transcribe_lock = threading.Lock()
    # Nudges hy model toward domain vocabulary (reduces letter/word errors on banking terms).
    _HY_BANKING_HINT = (
        "Հարց վարկերի, ավանդների, մասնաճյուղերի, տոկոսների մասին։ "
        "Ամերիաբանկ, Ameriabank, ԱԿԲԱ բանկ, ACBA, ԻԴԲանկ, IDBank, "
        "վարկ, ավանդ, ժամկետային ավանդ, ցպահանջ, մասնաճյուղ, բանկոմատ, Երևան, Հայաստան։"
    )

    def _transcribe_sync(audio_arr: np.ndarray, lang_code: str | None) -> str:
        def _run(*, use_vad: bool, beam: int) -> str:
            m = model_holder["model"]
            if m is None:
                raise RuntimeError("whisper model not ready")
            b = max(1, beam)
            segments, _info = m.transcribe(
                audio_arr,
                language=lang_code,
                vad_filter=use_vad,
                beam_size=b,
                best_of=b,
                patience=1.0 if b <= 1 else 1.1,
                # Slightly more tolerant of quiet / accented speech (fewer false "no speech" drops).
                no_speech_threshold=0.5,
                initial_prompt=_HY_BANKING_HINT,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            return "".join(s.text for s in segments).strip()

        with _transcribe_lock:
            rms = float(np.sqrt(np.mean(np.square(audio_arr)))) if audio_arr.size else 0.0
            # Fast path: low beam first (real-time PTT). Escalate beam only if we likely have speech.
            text = _run(use_vad=vad_filter, beam=beam_primary)
            if not text and vad_filter and rms > 0.002:
                logger.info("STT empty with VAD; retrying without VAD (rms=%.5f)", rms)
                text = _run(use_vad=False, beam=beam_primary)
            if not text and rms > 0.002 and beam_retry > beam_primary:
                logger.info("STT still empty; retry with wider beam=%s (rms=%.5f)", beam_retry, rms)
                text = _run(use_vad=False, beam=beam_retry)
            if not text and rms > 0.002:
                logger.info("STT still empty; final pass beam=1 without VAD (rms=%.5f)", rms)
                text = _run(use_vad=False, beam=1)
            return text

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
        if model_holder["error"]:
            return JSONResponse(
                {"error": "model_unavailable", "detail": model_holder["error"]},
                status_code=500,
            )
        if model_holder["model"] is None:
            return JSONResponse({"error": "model_loading"}, status_code=503)
        try:
            lang_code = lang if lang else None
            text = await asyncio.to_thread(_transcribe_sync, audio, lang_code)
            return {"text": text}
        except Exception as exc:
            logger.exception("Whisper transcribe failed")
            return JSONResponse(
                {"error": "transcribe_failed", "detail": str(exc)},
                status_code=500,
            )

    @app.get("/health")
    def health():
        if model_holder["error"]:
            return JSONResponse(
                {
                    "status": "error",
                    "service": "voice-http-stt",
                    "detail": model_holder["error"],
                },
                status_code=500,
            )
        if model_holder["model"] is None:
            return JSONResponse(
                {
                    "status": "loading",
                    "service": "voice-http-stt",
                    "model": args.model,
                },
                status_code=503,
            )
        return {
            "status": "ok",
            "service": "voice-http-stt",
            "model": args.model,
        }

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
