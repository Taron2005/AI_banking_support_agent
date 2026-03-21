import React, { useMemo, useState } from "react";
import { Room, RoomEvent, createLocalAudioTrack } from "livekit-client";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const container = {
  fontFamily: "Arial, sans-serif",
  maxWidth: 900,
  margin: "20px auto",
  padding: 16,
};

export default function App() {
  const [query, setQuery] = useState("");
  const [sessionId, setSessionId] = useState("react-demo-session");
  const [indexName, setIndexName] = useState("multi_model_index");
  const [verbose, setVerbose] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState(null);
  const [lkUrl, setLkUrl] = useState("ws://127.0.0.1:7880");
  const [lkToken, setLkToken] = useState("");
  const [lkConnected, setLkConnected] = useState(false);
  const [lkStatus, setLkStatus] = useState("Disconnected");
  const [voiceLog, setVoiceLog] = useState([]);
  const [roomRef, setRoomRef] = useState(null);

  const statusColor = useMemo(() => {
    if (!response) return "#666";
    return response.status === "answered" ? "#1b7f2a" : "#b02a37";
  }, [response]);

  async function askRuntime(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          query,
          index_name: indexName,
          top_k: 6,
          verbose,
        }),
      });
      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setError(String(err));
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
            `[${body.status}] ${body.answer_text || ""}`,
          ]);
        } catch {
          // ignore non-JSON packets
        }
      });
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === "audio") {
          track.attach();
          setVoiceLog((prev) => [...prev, "Assistant audio track subscribed"]);
        }
      });
      await room.connect(lkUrl, lkToken);
      const mic = await createLocalAudioTrack();
      await room.localParticipant.publishTrack(mic);
      setVoiceLog((prev) => [...prev, "Microphone published"]);
      setRoomRef(room);
    } catch (err) {
      setLkStatus(`LiveKit error: ${String(err)}`);
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
    <div style={container}>
      <h2>Voice AI Banking - React Demo</h2>
      <p>Minimal frontend for runtime manual testing.</p>

      <form onSubmit={askRuntime}>
        <label>Session ID</label>
        <input value={sessionId} onChange={(e) => setSessionId(e.target.value)} style={{ width: "100%", marginBottom: 8 }} />

        <label>Index Name</label>
        <input value={indexName} onChange={(e) => setIndexName(e.target.value)} style={{ width: "100%", marginBottom: 8 }} />

        <label>Question</label>
        <textarea
          rows={4}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about credit / deposit / branch"
          style={{ width: "100%", marginBottom: 8 }}
        />

        <label style={{ display: "block", marginBottom: 10 }}>
          <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} /> verbose trace
        </label>

        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "Asking..." : "Ask"}
        </button>
      </form>

      {error ? <p style={{ color: "#b02a37" }}>{error}</p> : null}

      {response ? (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ color: statusColor }}>Status: {response.status}</h3>
          <pre style={{ whiteSpace: "pre-wrap", background: "#f6f8fa", padding: 12 }}>
            {response.answer_text}
          </pre>
          <h4>Sources</h4>
          <ul>
            {(response.used_sources || []).map((u) => (
              <li key={u}>{u}</li>
            ))}
          </ul>
          <h4>Decision Trace</h4>
          <ul>
            {(response.decision_trace || []).map((t, i) => (
              <li key={`${i}-${t}`}>{t}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <hr style={{ margin: "24px 0" }} />
      <h3>LiveKit Voice (Mic Test)</h3>
      <p style={{ marginTop: 0 }}>
        Connect with self-hosted LiveKit token, then speak from microphone.
      </p>
      <label>LiveKit URL</label>
      <input
        value={lkUrl}
        onChange={(e) => setLkUrl(e.target.value)}
        style={{ width: "100%", marginBottom: 8 }}
      />
      <label>LiveKit Token</label>
      <textarea
        rows={3}
        value={lkToken}
        onChange={(e) => setLkToken(e.target.value)}
        style={{ width: "100%", marginBottom: 8 }}
      />
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button onClick={connectLiveKit} disabled={lkConnected || !lkToken.trim()}>
          Connect + Publish Mic
        </button>
        <button onClick={disconnectLiveKit} disabled={!lkConnected}>
          Disconnect
        </button>
      </div>
      <p>Status: {lkStatus}</p>
      <h4>Voice log</h4>
      <ul>
        {voiceLog.map((x, i) => (
          <li key={`${i}-${x}`}>{x}</li>
        ))}
      </ul>
    </div>
  );
}

