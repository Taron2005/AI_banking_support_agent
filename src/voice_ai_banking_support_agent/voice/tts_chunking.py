from __future__ import annotations

import re


def split_for_sequential_tts(
    text: str,
    *,
    max_chunk_chars: int = 380,
    min_chunk_chars: int = 10,
) -> list[str]:
    """
    Split Armenian (or mixed) assistant text into chunks for sequential TTS playout.

    Smaller first chunks reduce time-to-first-audio after the LLM returns. Chunks are
    merged when splits would be too short, and long paragraphs are split on spaces.
    """

    raw = (text or "").strip()
    if not raw:
        return []

    # Sentence-like breaks: Armenian full stop U+0589, Latin . ! ? … (spacing may vary)
    pieces = re.split(r"[\u0589.!?\u2026]+|\s*\n+\s*", raw)
    pieces = [p.strip() for p in pieces if p.strip()]
    if not pieces:
        return [raw]

    # One sentence (or line fragment) per chunk by default so the first TTS request stays small
    # and time-to-first-audio improves; only glue very short fragments to neighbors.
    merged: list[str] = []
    for p in pieces:
        if not merged:
            merged.append(p)
            continue
        if len(p) < min_chunk_chars:
            merged[-1] = f"{merged[-1]} {p}".strip()
        elif len(merged[-1]) < min_chunk_chars:
            merged[-1] = f"{merged[-1]} {p}".strip()
        else:
            merged.append(p)

    out: list[str] = []
    for m in merged:
        if len(m) < min_chunk_chars and out:
            out[-1] = f"{out[-1]} {m}".strip()
        else:
            out.append(m)

    # Hard cap: if a single segment still exceeds max, split by words
    final: list[str] = []
    for seg in out:
        if len(seg) <= max_chunk_chars:
            final.append(seg)
            continue
        words = seg.split()
        cur = ""
        for w in words:
            cand = w if not cur else f"{cur} {w}"
            if len(cand) <= max_chunk_chars:
                cur = cand
            else:
                if cur:
                    final.append(cur)
                cur = w
        if cur:
            final.append(cur)

    return final if final else [raw]
