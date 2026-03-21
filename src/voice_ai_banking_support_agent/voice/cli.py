from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from pathlib import Path

    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parents[3]
    load_dotenv(_root / ".env", override=False)
except ImportError:
    pass

from ..config import load_config
from ..runtime.llm_config import load_llm_settings
from ..runtime.runtime_config import load_runtime_settings
from ..runtime.session_state import SessionStateStore
from ..utils.logging import setup_logging
from .factory import build_runtime_for_voice, build_voice_dependencies
from .livekit_agent import LiveKitParticipantContext, LiveKitVoiceAgent
from .stt import HTTPWhisperSTTProvider, MockSTTProvider
from .tts import HTTPTTSProvider, MockTTSProvider
from .voice_config import load_voice_config
from .voice_models import STTInput

import logging

_voice_log = logging.getLogger(__name__)


def _resolve_voice_config_path(project_root: str, voice_config_path: str | None) -> Path | None:
    if not voice_config_path:
        return None
    p = Path(voice_config_path)
    return p if p.is_absolute() else Path(project_root).resolve() / p


def _log_llm_settings(llm_settings) -> None:
    """Match backend startup: make Gemini configuration obvious for voice turns."""

    if llm_settings.provider == "mock":
        _voice_log.warning(
            "LLM is MOCK — synthesis uses MockLLMClient in tests or explicit mock config; not for production demos."
        )
        return
    if llm_settings.provider == "gemini" and not llm_settings.is_live_llm_configured():
        _voice_log.error(
            "Gemini API key missing (GEMINI_API_KEY / GOOGLE_API_KEY). "
            "Voice answers will use explicit extractive fallback like the REST API without a key."
        )
        return
    _voice_log.info(
        "LLM (voice stack): provider=%s model=%s — same RuntimeOrchestrator as POST /chat",
        llm_settings.provider,
        llm_settings.model,
    )


def _log_voice_providers(voice_cfg, deps) -> None:
    """Make STT/TTS mode obvious at startup (mock vs HTTP is a common misconfiguration)."""
    stt_ep = (voice_cfg.stt.endpoint or "").strip()
    tts_ep = (voice_cfg.tts.endpoint or "").strip()
    if isinstance(deps.stt, MockSTTProvider):
        _voice_log.error(
            "STT is MOCK (no real speech-to-text). Set VOICE_STT_ENDPOINT in .env "
            "or stt.endpoint in voice_config.yaml, then restart this agent. "
            "Also run: python scripts/voice_http_stt_server.py"
        )
    elif isinstance(deps.stt, HTTPWhisperSTTProvider):
        _voice_log.info("STT: HTTP Whisper -> %s", stt_ep or "(missing - check .env)")
    if isinstance(deps.tts, MockTTSProvider):
        _voice_log.error(
            "TTS is MOCK (silent/minimal audio). Set VOICE_TTS_ENDPOINT in .env "
            "or tts.endpoint in voice_config.yaml. Run: python scripts/voice_http_tts_server.py"
        )
    elif isinstance(deps.tts, HTTPTTSProvider):
        _voice_log.info("TTS: HTTP -> %s", tts_ep or "(missing - check .env)")


def _safe(text: str) -> str:
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def run_voice_smoke(
    *,
    project_root: str,
    app_config_path: str | None,
    runtime_config_path: str | None,
    llm_config_path: str | None,
    voice_config_path: str | None,
    index_name: str,
) -> None:
    app_cfg = load_config(project_root=Path(project_root).resolve(), config_yaml=Path(app_config_path).resolve() if app_config_path else None)
    runtime_settings = load_runtime_settings(runtime_config_path)
    llm_settings = load_llm_settings(llm_config_path)
    _log_llm_settings(llm_settings)
    vpath = _resolve_voice_config_path(project_root, voice_config_path)
    voice_cfg = load_voice_config(vpath)
    runtime = build_runtime_for_voice(
        app_config=app_cfg, runtime_settings=runtime_settings, llm_settings=llm_settings
    )
    deps = build_voice_dependencies(voice_cfg)
    _log_voice_providers(voice_cfg, deps)
    agent = LiveKitVoiceAgent(
        runtime=runtime,
        state_store=SessionStateStore(),
        stt_provider=deps.stt,
        tts_provider=deps.tts,
        voice_config=voice_cfg,
    )
    participant = LiveKitParticipantContext(room_name=voice_cfg.livekit.room_name, participant_identity="smoke-user")
    samples = [
        "Ամերիաբանկը ինչ սպառողական վարկեր ունի",
        "իսկ դոլարով?",
        "Which bank is best for loans?",
    ]
    for s in samples:
        result = agent.process_turn(
            participant=participant,
            payload=STTInput(content=s.encode("utf-8"), encoding="text", language=voice_cfg.stt.language),
            index_name=index_name,
        )
        print(_safe(json.dumps(
            {
                "query": s,
                "status": result.runtime_response.status,
                "answer_text": result.runtime_response.answer_text,
                "refusal_reason": result.runtime_response.refusal_reason,
                "decision_trace": result.runtime_response.decision_trace,
                "tts_bytes": len(result.tts_output.audio),
            },
            ensure_ascii=False,
        )))


def run_livekit_agent(
    *,
    project_root: str,
    app_config_path: str | None,
    runtime_config_path: str | None,
    llm_config_path: str | None,
    voice_config_path: str | None,
    index_name: str,
) -> None:
    app_cfg = load_config(project_root=Path(project_root).resolve(), config_yaml=Path(app_config_path).resolve() if app_config_path else None)
    runtime_settings = load_runtime_settings(runtime_config_path)
    llm_settings = load_llm_settings(llm_config_path)
    _log_llm_settings(llm_settings)
    vpath = _resolve_voice_config_path(project_root, voice_config_path)
    voice_cfg = load_voice_config(vpath)
    runtime = build_runtime_for_voice(
        app_config=app_cfg, runtime_settings=runtime_settings, llm_settings=llm_settings
    )
    deps = build_voice_dependencies(voice_cfg)
    _log_voice_providers(voice_cfg, deps)
    agent = LiveKitVoiceAgent(
        runtime=runtime,
        state_store=SessionStateStore(),
        stt_provider=deps.stt,
        tts_provider=deps.tts,
        voice_config=voice_cfg,
    )
    agent.run_self_hosted(index_name=index_name)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Voice agent CLI")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--runtime-config", type=str, default=None)
    p.add_argument("--llm-config", type=str, default="llm_config.yaml")
    p.add_argument("--voice-config", type=str, default=None)
    p.add_argument("--index-name", type=str, required=True)
    p.add_argument("--mode", choices=["smoke", "livekit"], default="smoke")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    if args.mode == "smoke":
        run_voice_smoke(
            project_root=args.project_root,
            app_config_path=args.config,
            runtime_config_path=args.runtime_config,
            llm_config_path=args.llm_config,
            voice_config_path=args.voice_config,
            index_name=args.index_name,
        )
        return
    run_livekit_agent(
        project_root=args.project_root,
        app_config_path=args.config,
        runtime_config_path=args.runtime_config,
        llm_config_path=args.llm_config,
        voice_config_path=args.voice_config,
        index_name=args.index_name,
    )


if __name__ == "__main__":
    main()

