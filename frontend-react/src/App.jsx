import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Room, RoomEvent, createLocalAudioTrack } from "livekit-client";
import { MessageContent } from "./MessageContent.jsx";
import { VoiceLiveHud } from "./VoiceLiveHud.jsx";

const VOICE_DISCONNECTED = "disconnected";
const VOICE_IDLE = "idle";
const VOICE_LISTENING = "listening";
const VOICE_PROCESSING = "processing";
const VOICE_SPEAKING = "speaking";
const VOICE_ERROR = "error";

function voicePhaseLabel(phase, detail, recordSeconds) {
  switch (phase) {
    case VOICE_DISCONNECTED:
      return "Disconnected";
    case VOICE_IDLE:
      return "Ready — tap Mic, speak, then Stop & send";
    case VOICE_LISTENING: {
      const t = recordSeconds > 0 ? `${recordSeconds}s` : "…";
      return `Listening (${t}) — tap Stop & send when finished`;
    }
    case VOICE_PROCESSING:
      if (detail === "transcribing") return "Recognizing speech…";
      if (detail === "answering") return "Preparing answer (Gemini + evidence)…";
      return "Processing…";
    case VOICE_SPEAKING:
      return "Assistant speaking…";
    case VOICE_ERROR:
      return "Error";
    default:
      return "…";
  }
}

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? "" : "http://127.0.0.1:8000");
const LIVEKIT_IDENTITY = import.meta.env.VITE_LIVEKIT_IDENTITY || "web-user-1";
/** Must match voice agent + backend LiveKit room so text /chat and voice share SessionState. */
const LIVEKIT_ROOM = import.meta.env.VITE_LIVEKIT_ROOM || "banking-support-room";
const VOICE_ALIGNED_SESSION_ID = `lk::${LIVEKIT_ROOM}::${LIVEKIT_IDENTITY}`;

function normalizeLivekitWsUrl(url) {
  const u = (url || "").trim().replace(/\/+$/, "");
  if (!u) return u;
  const lower = u.toLowerCase();
  if (lower.startsWith("https://")) return `wss://${u.slice(8)}`;
  if (lower.startsWith("http://")) return `ws://${u.slice(7)}`;
  return u;
}

/** Align with server voice_config: self-hosted only, no LiveKit Cloud. */
function isRejectedLiveKitCloudUrl(url) {
  const u = (url || "").toLowerCase();
  return u.includes("livekit.cloud") || u.includes("cloud.livekit.io");
}

function voicePillClass(phase, connected) {
  if (phase === VOICE_ERROR) return "pill pill-voice error";
  if (!connected) return "pill pill-voice pill-neutral";
  if (phase === VOICE_LISTENING) return "pill pill-voice listening";
  if (phase === VOICE_SPEAKING) return "pill pill-voice speaking";
  if (phase === VOICE_PROCESSING) return "pill pill-voice pill-warn";
  return "pill pill-voice pill-ok";
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sessionId, setSessionId] = useState(VOICE_ALIGNED_SESSION_ID);
  const [verbose, setVerbose] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiOk, setApiOk] = useState(null);
  const [ready, setReady] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [lkUrl, setLkUrl] = useState(() => import.meta.env.VITE_LIVEKIT_URL || "");
  const [lkConnected, setLkConnected] = useState(false);
  const [lkBusy, setLkBusy] = useState(false);
  const [voicePhase, setVoicePhase] = useState(VOICE_DISCONNECTED);
  const [voiceDetail, setVoiceDetail] = useState("");
  const [voiceProcessingDetail, setVoiceProcessingDetail] = useState("");
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [voiceLog, setVoiceLog] = useState([]);
  const [micPreviewStream, setMicPreviewStream] = useState(null);
  const roomRef = useRef(null);
  const micTrackRef = useRef(null);
  const assistantAudioElRef = useRef(null);
  const bottomRef = useRef(null);
  const recordTimerRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (voicePhase === VOICE_LISTENING) {
      setRecordSeconds(0);
      recordTimerRef.current = window.setInterval(() => {
        setRecordSeconds((s) => s + 1);
      }, 1000);
    } else {
      if (recordTimerRef.current) {
        clearInterval(recordTimerRef.current);
        recordTimerRef.current = null;
      }
      setRecordSeconds(0);
    }
    return () => {
      if (recordTimerRef.current) {
        clearInterval(recordTimerRef.current);
        recordTimerRef.current = null;
      }
    };
  }, [voicePhase]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, r] = await Promise.all([
          fetch(`${API_BASE}/health`, { method: "GET" }),
          fetch(`${API_BASE}/ready`, { method: "GET" }),
        ]);
        if (!cancelled) {
          setApiOk(h.ok);
          if (r.ok) {
            const j = await r.json();
            setReady(j);
            if (!import.meta.env.VITE_LIVEKIT_URL && j.livekit_url) {
              setLkUrl(j.livekit_url);
            }
          }
        }
      } catch {
        if (!cancelled) setApiOk(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const apiPillClass = apiOk === null ? "pill pill-neutral" : apiOk ? "pill pill-ok" : "pill pill-off";
  const apiLabel = apiOk === null ? "API" : apiOk ? "API online" : "API offline";

  const llmOk =
    ready &&
    (ready.llm_configured === true || ready.llm_provider === "mock");
  const llmPillClass = !ready || llmOk ? "pill pill-ok" : "pill pill-warn";
  const llmLabel = !ready
    ? ""
    : ready.llm_provider === "mock"
      ? "LLM mock"
      : llmOk
        ? `Gemini OK (${ready.llm_model || ready.llm_provider || "?"})`
        : "Gemini key missing";

  async function sendMessage(e) {
    e?.preventDefault();
    const text = draft.trim();
    if (!text || loading) return;
    setError("");
    setDraft("");
    const userMsg = { role: "user", content: text, t: Date.now() };
    setMessages((m) => [...m, userMsg]);
    setLoading(true);
    const t0 = performance.now();
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          query: text,
          index_name: "hy_model_index",
          top_k: 8,
          verbose,
        }),
      });
      const bodyText = await res.text();
      let data;
      try {
        data = JSON.parse(bodyText);
      } catch {
        throw new Error(`Invalid JSON (${res.status}): ${bodyText.slice(0, 200)}`);
      }
      if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const ms = Math.round(performance.now() - t0);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer_text || "",
          status: data.status,
          sources: data.used_sources || [],
          trace: data.decision_trace || [],
          refusal: data.refusal_reason,
          answerSynthesis: data.answer_synthesis || null,
          llmError: data.llm_error || null,
          ms,
          t: Date.now(),
        },
      ]);
    } catch (err) {
      setError(String(err.message || err));
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: "Չհաջողվեց կապվել API-ի հետ։ Ստուգեք՝ աշխատում է սերվերը։",
          status: "error",
          t: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  const connectLiveKit = useCallback(async () => {
    if (lkConnected || lkBusy) return;
    setLkBusy(true);
    setVoicePhase(VOICE_DISCONNECTED);
    setVoiceDetail("Connecting…");
    try {
      let wsUrl =
        normalizeLivekitWsUrl(lkUrl) || normalizeLivekitWsUrl(ready?.livekit_url);
      if (!wsUrl) {
        const r = await fetch(`${API_BASE}/api/livekit/config`);
        const bodyText = await r.text();
        if (!r.ok) {
          throw new Error(
            `LiveKit config HTTP ${r.status}: ${bodyText.slice(0, 120) || r.statusText}`.trim()
          );
        }
        let cfg;
        try {
          cfg = JSON.parse(bodyText);
        } catch {
          throw new Error("LiveKit config was not JSON (wrong process on :8000?).");
        }
        wsUrl = normalizeLivekitWsUrl(cfg.livekit_url);
        if (!wsUrl) throw new Error("API did not return livekit_url.");
        setLkUrl(wsUrl);
      }
      if (!/^wss?:\/\//i.test(wsUrl)) {
        throw new Error(`Invalid LiveKit URL (need ws:// or wss://): ${wsUrl}`);
      }
      const tokRes = await fetch(
        `${API_BASE}/api/livekit/token?identity=${encodeURIComponent(LIVEKIT_IDENTITY)}`
      );
      if (!tokRes.ok) {
        let detail = await tokRes.text();
        try {
          const j = JSON.parse(detail);
          detail = j.detail || detail;
        } catch {
          /* keep */
        }
        throw new Error(detail || `token HTTP ${tokRes.status}`);
      }
      const { token } = await tokRes.json();
      if (!token) throw new Error("Empty token from API");

      const room = new Room();
      room.on(RoomEvent.Connected, () => {
        setLkConnected(true);
        setVoicePhase(VOICE_IDLE);
        setVoiceDetail("");
        setVoiceProcessingDetail("");
        setVoiceLog((prev) => [...prev, `${new Date().toISOString()} Room connected`]);
      });
      room.on(RoomEvent.Disconnected, () => {
        setLkConnected(false);
        setVoicePhase(VOICE_DISCONNECTED);
        setVoiceDetail("");
        setVoiceProcessingDetail("");
        roomRef.current = null;
      });
      room.on(RoomEvent.DataReceived, (payload, _participant, _kind, topic) => {
        try {
          const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
          const text = new TextDecoder().decode(bytes);
          const t = topic || "";
          if (t === "voice.state") {
            const j = JSON.parse(text);
            const st = j.state;
            const det = j.detail ? String(j.detail) : "";
            if (st === "idle") {
              setVoicePhase(VOICE_IDLE);
              setVoiceDetail("");
              setVoiceProcessingDetail("");
            } else if (st === "listening") {
              setVoicePhase(VOICE_LISTENING);
              setVoiceDetail("");
              setVoiceProcessingDetail("");
            } else if (st === "processing") {
              setVoicePhase(VOICE_PROCESSING);
              setVoiceProcessingDetail(det);
              setVoiceDetail("");
            } else if (st === "speaking") {
              setVoicePhase(VOICE_SPEAKING);
              setVoiceDetail("");
              setVoiceProcessingDetail("");
            } else if (st === "error") {
              setVoicePhase(VOICE_ERROR);
              const d = String(j.detail || "unknown_error");
              const errHy = {
                audio_too_short:
                  "Ձայնի նմուշը շատ կարճ էր (երկրորդ սեղմումից հետո հաճախակի)։ Սպասեք մի պահ Stop-ից հետո, խոսեք մի քանի վայրկյան, կամ ավելացրեք VOICE_PCM_TRAIL_PAUSE_SECONDS (~0.35) voice agent-ում։",
                stt_service_missing:
                  "STT չի կապված։ Գործարկեք voice_http_stt_server (պորտ 8088) և VOICE_STT_ENDPOINT .env-ում։",
                stt_failed: "STT սերվերի սխալ։ Ստուգեք voice agent լոգը։",
                stt_empty: "Խոսքը չճանաչվեց — կրկնեք ավելի հստակ։",
                tts_failed: "Ձայնային պատասխանի սինթեզը ձախողվեց։ Ստուգեք VOICE_TTS_ENDPOINT։",
                tts_playout_failed: "Ձայնի նվագարկումը ձախողվեց։",
                processing_failed: "Մշակման սխալ (RAG/Gemini)։ Ստուգեք voice agent լոգը։",
              };
              setVoiceDetail(errHy[d] || d);
              setVoiceProcessingDetail("");
            } else if (st === "busy") {
              setVoiceLog((p) => [...p, "Server busy — wait for assistant audio to finish."]);
              setVoicePhase(VOICE_IDLE);
              setVoiceDetail("Assistant is still answering — try again in a moment.");
              setVoiceProcessingDetail("");
            }
            return;
          }
          if (t === "voice.transcript.final") {
            const j = JSON.parse(text);
            const tx = String(j.text || "").trim();
            if (tx) {
              setMessages((m) => [
                ...m,
                {
                  role: "user",
                  content: tx,
                  t: Date.now(),
                  viaVoice: true,
                  voiceTranscript: true,
                },
              ]);
            }
            return;
          }
          if (t === "assistant.text.delta") {
            const j = JSON.parse(text);
            const frag = String(j.text || "");
            if (!frag) return;
            setMessages((m) => {
              const last = m[m.length - 1];
              if (last && last.role === "assistant" && last.streaming) {
                return [
                  ...m.slice(0, -1),
                  { ...last, content: last.content + frag, t: Date.now() },
                ];
              }
              return [
                ...m,
                {
                  role: "assistant",
                  content: frag,
                  streaming: true,
                  t: Date.now(),
                  viaVoice: true,
                },
              ];
            });
            return;
          }
          if (t === "assistant.text") {
            const body = JSON.parse(text);
            const ans = body.answer_text || "";
            if (ans) {
              setMessages((m) => {
                const last = m[m.length - 1];
                if (last && last.role === "assistant" && last.streaming) {
                  return [
                    ...m.slice(0, -1),
                    {
                      role: "assistant",
                      content: ans,
                      status: body.status,
                      sources: [],
                      trace: body.decision_trace || [],
                      refusal: body.refusal_reason,
                      answerSynthesis: body.answer_synthesis || null,
                      llmError: body.llm_error || null,
                      t: Date.now(),
                      viaVoice: true,
                      streamed: Boolean(body.streamed),
                    },
                  ];
                }
                return [
                  ...m,
                  {
                    role: "assistant",
                    content: ans,
                    status: body.status,
                    sources: [],
                    trace: body.decision_trace || [],
                    refusal: body.refusal_reason,
                    answerSynthesis: body.answer_synthesis || null,
                    llmError: body.llm_error || null,
                    t: Date.now(),
                    viaVoice: true,
                    streamed: Boolean(body.streamed),
                  },
                ];
              });
            }
            setVoiceLog((p) => [...p, `assistant.text ${body.status || ""}`]);
            return;
          }
        } catch {
          /* ignore */
        }
      });
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === "audio") {
          const el = track.attach();
          el.style.display = "none";
          document.body.appendChild(el);
          assistantAudioElRef.current = el;
          el.addEventListener("ended", () => {
            setVoicePhase((p) => (p === VOICE_SPEAKING ? VOICE_IDLE : p));
          });
          el.play?.().catch(() => {});
          setVoiceLog((p) => [...p, "Subscribed assistant audio"]);
        }
      });
      await room.connect(wsUrl, token, {
        autoSubscribe: true,
        disconnectOnPageLeave: true,
        peerConnectionTimeout: 60_000,
        websocketTimeout: 30_000,
        rtcConfig: {
          iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        },
      });
      roomRef.current = room;
    } catch (err) {
      const msg = String(err.message || err);
      setVoicePhase(VOICE_ERROR);
      setVoiceDetail(msg);
      setLkConnected(false);
      roomRef.current = null;
    } finally {
      setLkBusy(false);
    }
  }, [API_BASE, LIVEKIT_IDENTITY, lkConnected, lkBusy, lkUrl, ready]);

  const startVoiceTurn = useCallback(async () => {
    const room = roomRef.current;
    if (!room || voicePhase !== VOICE_IDLE) return;
    setVoiceDetail("");
    try {
      // After unpublishTrack(), reusing the same LocalAudioTrack often leaves mediaStreamTrack
      // ended or silent — the level meter + WebAudio analyser see no samples on turn 2+.
      let track = micTrackRef.current;
      const tr = track?.mediaStreamTrack;
      if (!track || !tr || tr.readyState === "ended") {
        track = await createLocalAudioTrack({
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        });
        micTrackRef.current = track;
      }
      // Publish mic before PTT "start" so the agent's LiveKit subscription exists
      // before buffering (avoids empty second-turn audio when consumer restarts).
      await room.localParticipant.publishTrack(track);
      // Give the agent time to subscribe before PTT start (second turn republish race).
      await new Promise((r) => setTimeout(r, 180));
      const enc = new TextEncoder();
      await room.localParticipant.publishData(
        enc.encode(
          JSON.stringify({
            type: "start",
            participant_identity: room.localParticipant.identity,
            session_id: sessionId,
          }),
        ),
        {
          reliable: true,
          topic: "voice.ptt",
        },
      );
      const msTrack = track.mediaStreamTrack;
      const prevStream =
        msTrack && msTrack.readyState !== "ended"
          ? track.mediaStream || new MediaStream([msTrack])
          : null;
      setMicPreviewStream(prevStream);
      setVoicePhase(VOICE_LISTENING);
      setVoiceProcessingDetail("");
    } catch (e) {
      const t = micTrackRef.current;
      if (t) {
        try {
          t.stop();
        } catch {
          /* ignore */
        }
        micTrackRef.current = null;
      }
      setMicPreviewStream(null);
      setVoicePhase(VOICE_ERROR);
      setVoiceDetail(String(e.message || e));
    }
  }, [voicePhase, sessionId]);

  const stopVoiceTurn = useCallback(async () => {
    const room = roomRef.current;
    if (!room || voicePhase !== VOICE_LISTENING) return;
    setVoicePhase(VOICE_PROCESSING);
    try {
      const enc = new TextEncoder();
      await room.localParticipant.publishData(
        enc.encode(
          JSON.stringify({
            type: "end",
            participant_identity: room.localParticipant.identity,
            session_id: sessionId,
          }),
        ),
        {
          reliable: true,
          topic: "voice.ptt",
        },
      );
      await new Promise((r) => setTimeout(r, 220));
      const track = micTrackRef.current;
      if (track) {
        try {
          await room.localParticipant.unpublishTrack(track);
        } finally {
          try {
            track.stop();
          } catch {
            /* ignore */
          }
          micTrackRef.current = null;
        }
      }
      setMicPreviewStream(null);
    } catch (e) {
      setMicPreviewStream(null);
      setVoicePhase(VOICE_ERROR);
      setVoiceDetail(String(e.message || e));
    }
  }, [voicePhase, sessionId]);

  async function disconnectLiveKit() {
    const room = roomRef.current;
    if (room) {
      const track = micTrackRef.current;
      if (track && room.localParticipant) {
        try {
          await room.localParticipant.unpublishTrack(track);
        } catch {
          /* ignore */
        }
        try {
          track.stop();
        } catch {
          /* ignore */
        }
      }
      await room.disconnect();
      roomRef.current = null;
    }
    micTrackRef.current = null;
    assistantAudioElRef.current = null;
    setMicPreviewStream(null);
    setLkConnected(false);
    setVoicePhase(VOICE_DISCONNECTED);
    setVoiceDetail("");
    setVoiceProcessingDetail("");
  }

  const voiceStatusLine = useMemo(() => {
    const base = voicePhaseLabel(voicePhase, voiceProcessingDetail, recordSeconds);
    if (voiceDetail && voicePhase === VOICE_ERROR) {
      return `${base}: ${voiceDetail}`;
    }
    if (voiceDetail && voicePhase === VOICE_IDLE) {
      return `${base} (${voiceDetail})`;
    }
    return base;
  }, [voicePhase, voiceDetail, voiceProcessingDetail, recordSeconds]);

  const micDisabled =
    !lkConnected || lkBusy || voicePhase === VOICE_PROCESSING || voicePhase === VOICE_SPEAKING;

  const handleMicButton = useCallback(() => {
    if (voicePhase === VOICE_ERROR) {
      setVoicePhase(VOICE_IDLE);
      setVoiceDetail("");
      setVoiceProcessingDetail("");
      return;
    }
    if (voicePhase === VOICE_LISTENING) stopVoiceTurn();
    else if (voicePhase === VOICE_IDLE) startVoiceTurn();
  }, [voicePhase, startVoiceTurn, stopVoiceTurn]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <div className="app-brand-title">Բանկային աջակցություն</div>
          <div className="app-brand-sub">
            Հայերեն RAG · hy_model_index · Gemini · LiveKit · push-to-talk + կենդանի ձայնի ցուցադրում
          </div>
        </div>
        <div className="pill-row">
          <span className={apiPillClass}>{apiLabel}</span>
          {ready ? <span className={llmPillClass}>{llmLabel}</span> : null}
          <span className={voicePillClass(voicePhase, lkConnected)} title={voiceDetail || undefined}>
            {voiceStatusLine}
          </span>
          <button
            type="button"
            className="btn btn-primary"
            onClick={connectLiveKit}
            disabled={!apiOk || lkConnected || lkBusy}
          >
            {lkBusy ? "…" : "Connect voice"}
          </button>
          <button
            type="button"
            className={`btn btn-mic ${voicePhase === VOICE_LISTENING ? "listen" : "idle"}`}
            onClick={handleMicButton}
            disabled={micDisabled && voicePhase !== VOICE_ERROR}
          >
            {voicePhase === VOICE_LISTENING
              ? "Stop & send"
              : voicePhase === VOICE_ERROR
                ? "Reset"
                : "Mic"}
          </button>
          <button type="button" className="btn btn-ghost" onClick={disconnectLiveKit} disabled={!lkConnected}>
            Disconnect
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => setShowAdvanced((s) => !s)}>
            {showAdvanced ? "Hide settings" : "Settings"}
          </button>
        </div>
      </header>

      {error ? (
        <div className="banner-error" role="alert">
          {error}
        </div>
      ) : null}

      {showAdvanced ? (
        <div className="panel-advanced">
          <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ color: "var(--text-muted)" }}>Session ID</span>
            <input
              className="composer-input"
              style={{ minHeight: 40 }}
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
            />
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12 }}>
            <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} />
            Show debug trace in chat
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 12 }}>
            <span style={{ color: "var(--text-muted)" }}>LiveKit URL</span>
            <input
              className="composer-input"
              style={{ minHeight: 40 }}
              value={lkUrl}
              onChange={(e) => setLkUrl(e.target.value)}
              placeholder="ws://127.0.0.1:7880"
            />
          </label>
          {voiceLog.length > 0 ? (
            <div style={{ marginTop: 14, maxHeight: 140, overflow: "auto", fontSize: 12, color: "var(--text-muted)" }}>
              <strong style={{ color: "var(--text)" }}>Voice log</strong>
              <ul style={{ margin: "8px 0 0 18px" }}>
                {voiceLog.slice(-12).map((line, i) => (
                  <li key={i}>{line}</li>
                ))}
              </ul>
            </div>
          ) : null}
          <p style={{ opacity: 0.65, margin: "14px 0 0", fontSize: 12 }}>
            API <code>{API_BASE || "(proxy)"}</code> · identity <code>{LIVEKIT_IDENTITY}</code>
          </p>
        </div>
      ) : null}

      {lkConnected ? (
        <div className="voice-hud-wrap">
          <VoiceLiveHud active={voicePhase === VOICE_LISTENING} mediaStream={micPreviewStream} />
        </div>
      ) : null}

      <main className="chat-main" style={{ display: "flex", flexDirection: "column" }}>
        {messages.length === 0 ? (
          <div className="hero-empty">
            <h2>Ուղեցույց</h2>
            <ol>
              <li>Հարցրեք վարկերի, ավանդների կամ մասնաճյուղերի մասին տեքստով (ստորև)։</li>
              <li>
                Կամ միացրեք ձայնը՝ <strong>Connect voice</strong>, ապա <strong>Mic</strong> — մակարդակի ցուցիչը ցույց է տալիս
                միկրոֆոնը։ <strong>Stop &amp; send</strong>-ից հետո ճանաչված տեքստը կերևա chat-ում <strong>STT</strong> պիտակով
                (Whisper), պատասխանը՝ TTS-ով։
              </li>
              <li>Պատասխանները հիմնված են միայն բանկերի պաշտոնական տվյալների վրա (hy_model_index + Gemini)։</li>
            </ol>
          </div>
        ) : null}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={`u-${i}-${m.t}`} className="bubble-user">
              {m.content}
              {m.voiceTranscript ? (
                <span className="voice-badge" title="Speech recognized (STT)">
                  STT
                </span>
              ) : null}
              {m.viaVoice && !m.voiceTranscript ? <span className="voice-badge">voice</span> : null}
            </div>
          ) : (
            <div
              key={`a-${i}-${m.t}`}
              className={`bubble-assistant${m.streaming ? " bubble-assistant-streaming" : ""}`}
            >
              <MessageContent text={m.content} />
              {m.streaming ? (
                <span className="stream-cursor" aria-hidden>
                  ▍
                </span>
              ) : null}
              {m.status && m.status !== "error" ? (
                <div className="bubble-meta">
                  {m.status}
                  {m.ms != null ? ` · ${m.ms} ms` : null}
                  {m.refusal ? ` · ${m.refusal}` : null}
                  {m.answerSynthesis === "llm"
                    ? " · AI synthesis (Gemini)"
                    : m.answerSynthesis === "extractive_fallback"
                      ? " · fallback (no LLM)"
                      : m.answerSynthesis === "extractive_only"
                        ? " · extractive mode"
                        : null}
                </div>
              ) : null}
              {m.llmError && verbose ? (
                <div className="bubble-meta" style={{ color: "var(--warn, #c9a227)" }}>
                  LLM: {m.llmError}
                </div>
              ) : null}
              {m.sources && m.sources.length > 0 ? (
                <div className="bubble-meta" style={{ marginTop: 10, textTransform: "none", letterSpacing: 0 }}>
                  Աղբյուրներ՝
                  <ul style={{ margin: "6px 0 0 18px" }}>
                    {m.sources.map((u) => (
                      <li key={u}>
                        <a href={u} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>
                          {u}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {verbose && m.trace && m.trace.length > 0 ? (
                <details className="bubble-meta" style={{ marginTop: 10, textTransform: "none" }}>
                  <summary style={{ cursor: "pointer" }}>Trace</summary>
                  <ul style={{ margin: "6px 0 0 18px" }}>
                    {m.trace.map((t, j) => (
                      <li key={j}>{t}</li>
                    ))}
                  </ul>
                </details>
              ) : null}
            </div>
          ),
        )}
        {loading ? (
          <div className="bubble-assistant" style={{ opacity: 0.85 }}>
            <span className="typing-dots">Մտածում եմ</span>
          </div>
        ) : null}
        <div ref={bottomRef} />
      </main>

      <form className="composer-bar" onSubmit={sendMessage}>
        <div className="composer-inner">
          <textarea
            className="composer-input"
            rows={2}
            placeholder="Հարց հայերենով…"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
          />
          <button
            type="submit"
            className="btn btn-primary"
            style={{ minWidth: 100 }}
            disabled={loading || !draft.trim()}
          >
            Ուղարկել
          </button>
        </div>
      </form>
    </div>
  );
}
