import React, { useEffect, useMemo, useRef, useState } from "react";
import { Room, RoomEvent, createLocalAudioTrack } from "livekit-client";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const shell = {
  minHeight: "100vh",
  background: "#0f1419",
  color: "#e7e9ea",
  fontFamily:
    'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  display: "flex",
  flexDirection: "column",
};

const header = {
  padding: "12px 20px",
  borderBottom: "1px solid #2f3336",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 12,
  flexWrap: "wrap",
};

const main = {
  flex: 1,
  display: "flex",
  maxWidth: 900,
  width: "100%",
  margin: "0 auto",
  flexDirection: "column",
  padding: "16px 20px 100px",
  overflowY: "auto",
};

const bubbleUser = {
  alignSelf: "flex-end",
  maxWidth: "85%",
  background: "#1d9bf0",
  color: "#fff",
  padding: "10px 14px",
  borderRadius: 16,
  borderBottomRightRadius: 4,
  marginBottom: 12,
  whiteSpace: "pre-wrap",
  lineHeight: 1.45,
};

const bubbleAsst = {
  alignSelf: "flex-start",
  maxWidth: "90%",
  background: "#202327",
  padding: "10px 14px",
  borderRadius: 16,
  borderBottomLeftRadius: 4,
  marginBottom: 12,
  whiteSpace: "pre-wrap",
  lineHeight: 1.45,
  border: "1px solid #2f3336",
};

const meta = {
  fontSize: 12,
  opacity: 0.65,
  marginTop: 6,
};

const composer = {
  position: "fixed",
  bottom: 0,
  left: 0,
  right: 0,
  padding: "12px 16px 20px",
  background: "linear-gradient(transparent, #0f1419 25%)",
  borderTop: "1px solid #2f3336",
};

const composerInner = {
  maxWidth: 900,
  margin: "0 auto",
  display: "flex",
  gap: 8,
  alignItems: "flex-end",
};

const inputStyle = {
  flex: 1,
  minHeight: 48,
  maxHeight: 160,
  padding: "12px 14px",
  borderRadius: 12,
  border: "1px solid #38444d",
  background: "#16181c",
  color: "#e7e9ea",
  fontSize: 15,
  resize: "vertical",
  outline: "none",
};

const btnPrimary = {
  padding: "12px 20px",
  borderRadius: 9999,
  border: "none",
  background: "#1d9bf0",
  color: "#fff",
  fontWeight: 600,
  cursor: "pointer",
  minHeight: 48,
};

const btnGhost = {
  padding: "8px 14px",
  borderRadius: 8,
  border: "1px solid #38444d",
  background: "transparent",
  color: "#e7e9ea",
  cursor: "pointer",
  fontSize: 13,
};

const panel = {
  marginTop: 12,
  padding: 12,
  borderRadius: 10,
  background: "#16181c",
  border: "1px solid #2f3336",
  fontSize: 13,
};

export default function App() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sessionId, setSessionId] = useState("web-session");
  const [indexName, setIndexName] = useState("hy_model_index");
  const [verbose, setVerbose] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiOk, setApiOk] = useState(null);
  const [showConfig, setShowConfig] = useState(false);
  const [lkUrl, setLkUrl] = useState("ws://127.0.0.1:7880");
  const [lkToken, setLkToken] = useState("");
  const [lkConnected, setLkConnected] = useState(false);
  const [lkStatus, setLkStatus] = useState("Disconnected");
  const [voiceLog, setVoiceLog] = useState([]);
  const [roomRef, setRoomRef] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/health`, { method: "GET" })
      .then((r) => {
        if (!cancelled) setApiOk(r.ok);
      })
      .catch(() => {
        if (!cancelled) setApiOk(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const apiLabel = useMemo(() => {
    if (apiOk === null) return "API: checking…";
    if (apiOk) return "API: reachable";
    return "API: offline";
  }, [apiOk]);

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
          index_name: indexName,
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
          content: `Չհաջողվեց կապվել API-ի հետ։ Ստուգեք՝ աշխատում է \`${API_BASE}\` սերվերը։`,
          status: "error",
          t: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function connectLiveKit() {
    try {
      const room = new Room();
      room.on(RoomEvent.Connected, () => {
        setLkConnected(true);
        setLkStatus("Connected");
      });
      room.on(RoomEvent.Disconnected, () => {
        setLkConnected(false);
        setLkStatus("Disconnected");
      });
      room.on(RoomEvent.DataReceived, (payload) => {
        try {
          const text = new TextDecoder().decode(payload);
          const body = JSON.parse(text);
          setVoiceLog((prev) => [
            ...prev,
            `[${body.status}] ${(body.answer_text || "").slice(0, 200)}`,
          ]);
        } catch {
          /* ignore */
        }
      });
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === "audio") {
          track.attach();
          setVoiceLog((prev) => [...prev, "Assistant audio subscribed"]);
        }
      });
      await room.connect(lkUrl, lkToken);
      const mic = await createLocalAudioTrack();
      await room.localParticipant.publishTrack(mic);
      setVoiceLog((prev) => [...prev, "Microphone published"]);
      setRoomRef(room);
    } catch (err) {
      setLkStatus(`Error: ${String(err)}`);
      setLkConnected(false);
    }
  }

  async function disconnectLiveKit() {
    if (roomRef) {
      await roomRef.disconnect();
      setRoomRef(null);
    }
    setLkConnected(false);
    setLkStatus("Disconnected");
  }

  return (
    <div style={shell}>
      <header style={header}>
        <div>
          <strong>Banking support</strong>
          <span style={{ opacity: 0.6, marginLeft: 8, fontSize: 14 }}>
            Հայերեն · վարկ / ավանդ / մասնաճյուղ
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span
            style={{
              fontSize: 13,
              padding: "4px 10px",
              borderRadius: 999,
              background: apiOk ? "#063d2f" : apiOk === false ? "#4a1520" : "#2a2f36",
            }}
          >
            {apiLabel}
          </span>
          <span
            style={{
              fontSize: 13,
              padding: "4px 10px",
              borderRadius: 999,
              background: lkConnected ? "#063d2f" : "#2a2f36",
            }}
          >
            LiveKit: {lkStatus}
          </span>
          <button type="button" style={btnGhost} onClick={() => setShowConfig((s) => !s)}>
            {showConfig ? "Hide config" : "Config"}
          </button>
        </div>
      </header>

      {showConfig ? (
        <div style={{ ...panel, margin: "12px auto 0", maxWidth: 900, width: "calc(100% - 32px)" }}>
          <div style={{ display: "grid", gap: 10, gridTemplateColumns: "1fr 1fr" }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              Session ID
              <input
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                style={{ ...inputStyle, minHeight: 40 }}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              Index name
              <input
                value={indexName}
                onChange={(e) => setIndexName(e.target.value)}
                style={{ ...inputStyle, minHeight: 40 }}
              />
            </label>
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10 }}>
            <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} />
            Verbose decision trace
          </label>
          <p style={{ opacity: 0.7, margin: "10px 0 0", fontSize: 12 }}>
            API base: <code>{API_BASE}</code> — override with <code>VITE_API_BASE_URL</code>.
          </p>
          <hr style={{ borderColor: "#2f3336", margin: "14px 0" }} />
          <strong style={{ fontSize: 13 }}>LiveKit (optional)</strong>
          <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
            <input
              placeholder="LIVEKIT_URL"
              value={lkUrl}
              onChange={(e) => setLkUrl(e.target.value)}
              style={{ ...inputStyle, minHeight: 40 }}
            />
            <textarea
              placeholder="Paste LIVEKIT_TOKEN"
              rows={2}
              value={lkToken}
              onChange={(e) => setLkToken(e.target.value)}
              style={inputStyle}
            />
            <div style={{ display: "flex", gap: 8 }}>
              <button type="button" style={btnGhost} onClick={connectLiveKit} disabled={lkConnected || !lkToken.trim()}>
                Connect + mic
              </button>
              <button type="button" style={btnGhost} onClick={disconnectLiveKit} disabled={!lkConnected}>
                Disconnect
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <main style={main}>
        {messages.length === 0 ? (
          <p style={{ opacity: 0.65, marginTop: 24 }}>
            Հարցրեք վարկերի, ավանդների կամ մասնաճյուղերի մասին։ Օրինակ՝ «Ինչ ավանդներ ունի ACBA-ն»։
          </p>
        ) : null}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={`u-${i}-${m.t}`} style={bubbleUser}>
              {m.content}
            </div>
          ) : (
            <div key={`a-${i}-${m.t}`} style={bubbleAsst}>
              {m.content}
              {m.status && m.status !== "error" ? (
                <div style={meta}>
                  Status: <strong>{m.status}</strong>
                  {m.ms != null ? ` · ${m.ms} ms` : null}
                  {m.refusal ? ` · refusal: ${m.refusal}` : null}
                </div>
              ) : null}
              {m.sources && m.sources.length > 0 ? (
                <div style={{ ...meta, marginTop: 8 }}>
                  Sources:
                  <ul style={{ margin: "4px 0 0 18px" }}>
                    {m.sources.map((u) => (
                      <li key={u}>
                        <a href={u} target="_blank" rel="noreferrer" style={{ color: "#1d9bf0" }}>
                          {u}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {verbose && m.trace && m.trace.length > 0 ? (
                <details style={{ ...meta, marginTop: 8 }}>
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
          <div style={{ ...bubbleAsst, opacity: 0.8 }}>
            <span className="typing">Մտածում եմ…</span>
          </div>
        ) : null}
        {error ? <div style={{ color: "#f4212e", fontSize: 14, marginBottom: 8 }}>{error}</div> : null}
        <div ref={bottomRef} />
      </main>

      <form style={composer} onSubmit={sendMessage}>
        <div style={composerInner}>
          <textarea
            style={inputStyle}
            rows={2}
            placeholder="Հարց…"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
          />
          <button type="submit" style={{ ...btnPrimary, opacity: loading || !draft.trim() ? 0.5 : 1 }} disabled={loading || !draft.trim()}>
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
