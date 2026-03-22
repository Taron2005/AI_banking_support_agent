import React, { useEffect, useRef, useState } from "react";

/**
 * Live mic level visualization while recording.
 * Transcription is only from server Whisper STT (after Stop & send) — no browser SpeechRecognition,
 * so preview text always matches what RAG receives.
 */

function useMicLevels(active, mediaStream) {
  const [levels, setLevels] = useState(() => Array.from({ length: 24 }, () => 0));
  const ctxRef = useRef(null);
  const analyserRef = useRef(null);
  const srcRef = useRef(null);
  const rafRef = useRef(null);
  const dataRef = useRef(null);

  useEffect(() => {
    if (!active || !mediaStream) {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (srcRef.current) {
        try {
          srcRef.current.disconnect();
        } catch {
          /* ignore */
        }
        srcRef.current = null;
      }
      if (ctxRef.current) {
        try {
          ctxRef.current.close();
        } catch {
          /* ignore */
        }
        ctxRef.current = null;
      }
      analyserRef.current = null;
      setLevels(Array.from({ length: 24 }, () => 0));
      return;
    }

    let cancelled = false;
    (async () => {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      const ctx = new AC();
      if (cancelled) {
        try {
          await ctx.close();
        } catch {
          /* ignore */
        }
        return;
      }
      ctxRef.current = ctx;
      try {
        await ctx.resume();
      } catch {
        /* ignore */
      }
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.65;
      analyserRef.current = analyser;
      const src = ctx.createMediaStreamSource(mediaStream);
      srcRef.current = src;
      src.connect(analyser);
      const buf = new Uint8Array(analyser.frequencyBinCount);
      dataRef.current = buf;

      const tick = () => {
        if (!analyserRef.current || !dataRef.current) return;
        analyserRef.current.getByteFrequencyData(dataRef.current);
        const n = 24;
        const bin = Math.floor(dataRef.current.length / n);
        const next = [];
        for (let i = 0; i < n; i++) {
          let sum = 0;
          const start = i * bin;
          for (let j = 0; j < bin; j++) sum += dataRef.current[start + j];
          const v = sum / (bin * 255);
          next.push(Math.min(1, v * 2.8));
        }
        setLevels(next);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    })();

    return () => {
      cancelled = true;
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (srcRef.current) {
        try {
          srcRef.current.disconnect();
        } catch {
          /* ignore */
        }
        srcRef.current = null;
      }
      if (ctxRef.current) {
        const c = ctxRef.current;
        ctxRef.current = null;
        c.close().catch(() => {});
      }
      analyserRef.current = null;
      setLevels(Array.from({ length: 24 }, () => 0));
    };
  }, [active, mediaStream]);

  return levels;
}

export function VoiceLiveHud({ active, mediaStream }) {
  const levels = useMicLevels(active, mediaStream);

  return (
    <div className={`voice-live-hud ${active ? "voice-live-hud-on" : ""}`} aria-live="polite">
      <div className="voice-live-hud-header">
        <span className="voice-live-dot" aria-hidden />
        <span className="voice-live-title">{active ? "Գրանցում ենք ձայնը" : "Ձայնային ռեժիմ"}</span>
        <span
          className="voice-live-badge"
          title="Ճանաչումը կատարում է սերվերի Whisper-ը Stop & send-ից հետո — նույն տեքստը, ինչ chat-ում STT պիտակով"
        >
          միայն Whisper (սերվեր)
        </span>
      </div>

      <div className="voice-meter" role="img" aria-label="Միկրոֆոնի մակարդակ">
        {levels.map((v, i) => (
          <div key={i} className="voice-meter-bar-wrap">
            <div className="voice-meter-bar" style={{ transform: `scaleY(${0.08 + v * 0.92})` }} />
          </div>
        ))}
      </div>

      {active ? (
        <div className="voice-live-caption">
          <p className="voice-live-caption-placeholder">
            Խոսեք հայերենով։ Ճանաչված տեքստը կերևա chat-ում <strong>STT</strong> պիտակով Stop & send-ից հետո —
            դա է ուղարկվում հարցմանը։
          </p>
        </div>
      ) : (
        <p className="voice-live-hint">Սեղմեք Mic — սկսեք խոսել, ապա Stop &amp; send։</p>
      )}
    </div>
  );
}
