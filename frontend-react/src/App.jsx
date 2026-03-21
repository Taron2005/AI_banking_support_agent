import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Room, RoomEvent, createLocalAudioTrack } from "livekit-client";

const VOICE_DISCONNECTED = "disconnected";
const VOICE_IDLE = "idle";
const VOICE_LISTENING = "listening";
const VOICE_PROCESSING = "processing";
const VOICE_SPEAKING = "speaking";
const VOICE_ERROR = "error";

function voicePhaseLabel(phase) {
  switch (phase) {
    case VOICE_DISCONNECTED:
      return "Disconnected";
    case VOICE_IDLE:
      return "Ready — tap mic to speak";
    case VOICE_LISTENING:
      return "Listening… tap again to send";
    case VOICE_PROCESSING:
      return "Thinking…";
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

function normalizeLivekitWsUrl(url) {
  const u = (url || "").trim().replace(/\/+$/, "");
  if (!u) return u;
  const lower = u.toLowerCase();
  if (lower.startsWith("https://")) return `wss://${u.slice(8)}`;
  if (lower.startsWith("http://")) return `ws://${u.slice(7)}`;
  return u;
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
  const [sessionId, setSessionId] = useState("web-session");
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
  const [voiceLog, setVoiceLog] = useState([]);
  const roomRef = useRef(null);
  const micTrackRef = useRef(null);
  const assistantAudioElRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

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

  const groqPillClass =
    !ready || ready.groq_configured ? "pill pill-ok" : "pill pill-warn";
  const groqLabel = !ready ? "" : ready.groq_configured ? "Groq ready" : "Groq key missing";

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
        setVoiceLog((prev) => [...prev, `${new Date().toISOString()} Room connected`]);
      });
      room.on(RoomEvent.Disconnected, () => {
        setLkConnected(false);
        setVoicePhase(VOICE_DISCONNECTED);
        setVoiceDetail("");
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
            if (st === "idle") {
              setVoicePhase(VOICE_IDLE);
              setVoiceDetail("");
            } else if (st === "processing") {
              setVoicePhase(VOICE_PROCESSING);
              setVoiceDetail("");
            } else if (st === "speaking") {
              setVoicePhase(VOICE_SPEAKING);
              setVoiceDetail("");
            } else if (st === "error") {
              setVoicePhase(VOICE_ERROR);
              setVoiceDetail(String(j.detail || "unknown_error"));
            } else if (st === "busy") {
              setVoiceLog((p) => [...p, "Server busy (finish assistant turn first)."]);
              setVoicePhase(VOICE_IDLE);
            }
            return;
          }
          if (t === "assistant.text") {
            const body = JSON.parse(text);
            const ans = body.answer_text || "";
            const ut = body.user_text || "";
            if (ut) {
              setMessages((m) => [...m, { role: "user", content: ut, t: Date.now(), viaVoice: true }]);
            }
            if (ans) {
              setMessages((m) => [
                ...m,
                {
                  role: "assistant",
                  content: ans,
                  status: body.status,
                  sources: [],
                  trace: body.decision_trace || [],
                  refusal: body.refusal_reason,
                  t: Date.now(),
                  viaVoice: true,
                },
              ]);
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
      const enc = new TextEncoder();
      await room.localParticipant.publishData(enc.encode(JSON.stringify({ type: "start" })), {
        reliable: true,
        topic: "voice.ptt",
      });
      let track = micTrackRef.current;
      if (!track) {
        track = await createLocalAudioTrack({
          echoCancellation: true,
          noiseSuppression: true,
        });
        micTrackRef.current = track;
      }
      await room.localParticipant.publishTrack(track);
      setVoicePhase(VOICE_LISTENING);
    } catch (e) {
      setVoicePhase(VOICE_ERROR);
      setVoiceDetail(String(e.message || e));
    }
  }, [voicePhase]);

  const stopVoiceTurn = useCallback(async () => {
    const room = roomRef.current;
    if (!room || voicePhase !== VOICE_LISTENING) return;
    setVoicePhase(VOICE_PROCESSING);
    try {
      const enc = new TextEncoder();
      await room.localParticipant.publishData(enc.encode(JSON.stringify({ type: "end" })), {
        reliable: true,
        topic: "voice.ptt",
      });
      await new Promise((r) => setTimeout(r, 150));
      const track = micTrackRef.current;
      if (track) await room.localParticipant.unpublishTrack(track);
    } catch (e) {
      setVoicePhase(VOICE_ERROR);
      setVoiceDetail(String(e.message || e));
    }
  }, [voicePhase]);

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
      }
      await room.disconnect();
      roomRef.current = null;
    }
    micTrackRef.current = null;
    assistantAudioElRef.current = null;
    setLkConnected(false);
    setVoicePhase(VOICE_DISCONNECTED);
    setVoiceDetail("");
  }

  const voiceStatusLine = useMemo(() => {
    const base = voicePhaseLabel(voicePhase);
    if (voiceDetail && voicePhase === VOICE_ERROR) {
      return `${base}: ${voiceDetail}`;
    }
    return base;
  }, [voicePhase, voiceDetail]);

  const micDisabled =
    !lkConnected || lkBusy || voicePhase === VOICE_PROCESSING || voicePhase === VOICE_SPEAKING;

  const handleMicButton = useCallback(() => {
    if (voicePhase === VOICE_ERROR) {
      setVoicePhase(VOICE_IDLE);
      setVoiceDetail("");
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
            Հայերեն RAG · hy_model_index · Groq · push-to-talk voice
          </div>
        </div>
        <div className="pill-row">
          <span className={apiPillClass}>{apiLabel}</span>
          {ready ? <span className={groqPillClass}>{groqLabel}</span> : null}
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

      <main className="chat-main" style={{ display: "flex", flexDirection: "column" }}>
        {messages.length === 0 ? (
          <div className="hero-empty">
            <h2>Ուղեցույց</h2>
            <ol>
              <li>Հարցրեք վարկերի, ավանդների կամ մասնաճյուղերի մասին տեքստով (ստորև)։</li>
              <li>
                Կամ միացրեք ձայնը՝ <strong>Connect voice</strong>, ապա <strong>Mic</strong> — խոսեք հայերեն, ապա{" "}
                <strong>Stop & send</strong>։
              </li>
              <li>Պատասխանները հիմնված են միայն բանկերի պաշտոնական տվյալների վրա (Groq + hy_model_index)։</li>
            </ol>
          </div>
        ) : null}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={`u-${i}-${m.t}`} className="bubble-user">
              {m.content}
              {m.viaVoice ? <span className="voice-badge">voice</span> : null}
            </div>
          ) : (
            <div key={`a-${i}-${m.t}`} className="bubble-assistant">
              {m.content}
              {m.status && m.status !== "error" ? (
                <div className="bubble-meta">
                  {m.status}
                  {m.ms != null ? ` · ${m.ms} ms` : null}
                  {m.refusal ? ` · ${m.refusal}` : null}
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
