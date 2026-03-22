import React, { useCallback, useEffect, useRef, useState } from "react";

/**
 * Live mic level visualization + optional browser SpeechRecognition captions (hy-AM).
 * Captions are preview-only; server STT remains authoritative for RAG.
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

function getSpeechRecognitionCtor() {
  return typeof window !== "undefined"
    ? window.SpeechRecognition || window.webkitSpeechRecognition || null
    : null;
}

function useBrowserSpeechCaption(active, lang) {
  const [interim, setInterim] = useState("");
  const [finalBuf, setFinalBuf] = useState("");
  const [speechError, setSpeechError] = useState("");
  const recRef = useRef(null);
  const activeRef = useRef(false);
  const langRef = useRef(lang);

  useEffect(() => {
    langRef.current = lang;
  }, [lang]);

  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  const supported = Boolean(getSpeechRecognitionCtor());

  const stopRec = useCallback(() => {
    const r = recRef.current;
    recRef.current = null;
    if (r) {
      try {
        r.onend = null;
        r.onerror = null;
        r.onresult = null;
        r.stop();
      } catch {
        /* ignore */
      }
    }
  }, []);

  useEffect(() => {
    if (!active) {
      stopRec();
      setInterim("");
      return;
    }
    if (!supported) {
      setSpeechError("");
      return;
    }

    const Ctor = getSpeechRecognitionCtor();
    const rec = new Ctor();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = langRef.current || "hy-AM";
    rec.maxAlternatives = 1;

    let finalText = "";
    setInterim("");
    setFinalBuf("");
    setSpeechError("");

    rec.onresult = (ev) => {
      let interimPiece = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const res = ev.results[i];
        const tx = (res[0] && res[0].transcript) || "";
        if (res.isFinal) finalText = `${finalText} ${tx}`.trim();
        else interimPiece += tx;
      }
      setFinalBuf(finalText.trim());
      setInterim(interimPiece.trim());
    };

    rec.onerror = (ev) => {
      const code = ev.error || "unknown";
      if (code === "aborted" || code === "no-speech") return;
      setSpeechError(String(code));
    };

    rec.onend = () => {
      if (!activeRef.current || recRef.current !== rec) return;
      window.setTimeout(() => {
        if (!activeRef.current || recRef.current !== rec) return;
        try {
          rec.start();
        } catch {
          /* ignore — may already be starting */
        }
      }, 90);
    };

    recRef.current = rec;
    try {
      rec.start();
    } catch (e) {
      setSpeechError(String(e.message || e));
    }

    return () => {
      stopRec();
      setInterim("");
      setFinalBuf("");
    };
  }, [active, supported, stopRec]);

  const displayLine = [finalBuf, interim].filter(Boolean).join(" ").trim();

  return { supported, displayLine, interim, finalBuf, speechError };
}

export function VoiceLiveHud({ active, mediaStream, speechLang = "hy-AM" }) {
  const levels = useMicLevels(active, mediaStream);
  const { supported, displayLine, speechError } = useBrowserSpeechCaption(active, speechLang);

  return (
    <div className={`voice-live-hud ${active ? "voice-live-hud-on" : ""}`} aria-live="polite">
      <div className="voice-live-hud-header">
        <span className="voice-live-dot" aria-hidden />
        <span className="voice-live-title">
          {active ? "Գրանցում ենք ձայնը" : "Ձայնային ռեժիմ"}
        </span>
        {!supported ? (
          <span className="voice-live-badge voice-live-badge-warn" title="Browser Speech API unavailable">
            մակարդակի ցուցիչ միայն
          </span>
        ) : (
          <span className="voice-live-badge" title="Նախադիտում է բրաուզերի ճանաչումը. Պաշտոնական տեքստը՝ սերվերի STT">
            կենդանի տեքստ (նախադիտում)
          </span>
        )}
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
          {displayLine ? (
            <p className="voice-live-caption-text">{displayLine}</p>
          ) : (
            <p className="voice-live-caption-placeholder">
              {supported
                ? "Խոսեք հայերեն — տեքստը կերևա այստեղ խոսելիս։"
                : "Խոսեք — սյուները պետք է շարժվեն, եթե միկրոֆոնը աշխատում է։"}
            </p>
          )}
          {speechError ? <p className="voice-live-caption-err">Ճանաչում՝ {speechError}</p> : null}
        </div>
      ) : (
        <p className="voice-live-hint">Սեղմեք Mic — սկսեք խոսել, ապա Stop &amp; send։</p>
      )}
    </div>
  );
}
