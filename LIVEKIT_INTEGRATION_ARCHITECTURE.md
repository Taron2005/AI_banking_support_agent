# LiveKit Integration Architecture

## Purpose

This phase adds a real-time voice transport layer around the existing runtime brain.

- **Runtime core (`runtime/`)**: topic control, retrieval grounding, evidence checks, refusals, answer text.
- **Voice layer (`voice/`)**: session mapping, STT/TTS provider abstraction, LiveKit transport loop.

## End-to-end flow

1. LiveKit participant sends turn (audio or text packet in local dev mode).
2. Voice layer maps `room + participant` to deterministic runtime `session_id`.
3. STT provider converts input payload to user text.
4. Voice layer calls runtime orchestrator (`RuntimeRequest`).
5. Runtime returns structured result (`answered/refused/clarify`).
6. Voice layer sends returned answer text to TTS.
7. TTS audio bytes are emitted back over transport.

## Separation of responsibilities

- Voice layer does **not** classify topic, decide refusal, or check evidence.
- Runtime layer remains the single decision-making authority.
- Voice layer only transports/transforms modalities.

## Session model

Runtime session ID format:

- `lk::<room_name>::<participant_identity>`

This preserves follow-up continuity per participant in the same room.

## STT/TTS abstraction

- `voice/stt.py` exposes `STTProvider` protocol.
- `voice/tts.py` exposes `TTSProvider` protocol.
- Default voice config targets **HTTP STT/TTS**; mock is used only when endpoints are unset or `VOICE_USE_MOCK=1`.
- Providers are built via `voice/factory.py`, so later replacements are isolated.

## LiveKit (self-hosted) assumptions

- Uses self-hosted server URL (`LIVEKIT_URL` / `voice_config.yaml`).
- Auth token expected via `LIVEKIT_TOKEN` in current implementation.
- LiveKit Cloud URLs are explicitly rejected at startup.
- This project uses open-source/self-hosted LiveKit only.

## Scalability and safety

- Runtime guardrails preserved unchanged.
- Provider/transport swappable.
- Decision traces from runtime can be passed through in debug mode.
