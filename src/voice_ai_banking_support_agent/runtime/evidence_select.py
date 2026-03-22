from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .models import RetrievedChunk

_NAV_LATIN = frozenset(
    "home menu search login signin signup register cookie cookies twitter facebook "
    "linkedin instagram youtube tiktok gdpr privacy policy terms newsletter subscribe "
    "sitemap contact us read more".split()
)
_NAV_HY = (
    "օգտվող",
    "մուտք",
    "փնտրել",
    "մենյու",
    "գլխավոր",
    "հետադարձ",
    "կարդալ ավելին",
    "բոլոր իրավունքները",
    "պայմաններ",
    "գաղտնիություն",
)


def normalize_http_url(url: str) -> str:
    u = url.strip()
    if not u:
        return u
    try:
        p = urlparse(u)
        if not p.scheme or not p.netloc:
            return u.rstrip("/")
        path = p.path or ""
        path = path.rstrip("/") or "/"
        q = parse_qsl(p.query, keep_blank_values=True)
        q = sorted((k, v) for k, v in q if k)
        query = urlencode(q) if q else ""
        return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", query, ""))
    except Exception:
        return u.split("#", 1)[0].rstrip("/")


def dedupe_urls(urls: list[str], *, max_n: int = 6) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in urls:
        n = normalize_http_url(raw)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(raw.strip())
        if len(out) >= max_n:
            break
    return out


_TOKEN_RE = re.compile(r"[\w\u0561-\u0587\u0531-\u0556]{2,}", re.UNICODE)


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


def query_term_overlap(query: str, text: str) -> float:
    qt, tt = _tokens(query), _tokens(text)
    if not qt:
        return 0.0
    return len(qt & tt) / max(len(qt), 1)


def content_substance_score(text: str) -> float:
    """Higher when text looks like prose / facts, lower for nav-like boilerplate."""

    t = text.strip()
    if len(t) < 35:
        return 0.12
    low = t.lower()
    score = 1.0
    nav_hits = sum(1 for w in _NAV_LATIN if w in low)
    nav_hits += sum(1 for w in _NAV_HY if w in low)
    if nav_hits >= 5:
        score *= 0.45
    elif nav_hits >= 3:
        score *= 0.65

    lines = [ln.strip() for ln in t.replace("\r", "\n").split("\n") if ln.strip()]
    if len(lines) >= 5:
        short = sum(1 for ln in lines if len(ln) < 18)
        if short / len(lines) > 0.58:
            score *= 0.5

    if re.search(r"\d", t):
        score *= 1.08
    if "%" in t or "տոկոս" in low or "rate" in low or "ֆիքսված" in low:
        score *= 1.06

    return max(0.06, min(1.0, score))


def _bank_key_chunk(c: RetrievedChunk) -> str:
    raw = (c.chunk.bank_key or c.chunk.bank_name or "").strip().lower()
    return raw or "_unknown"


def chunk_matches_bank_keys(c: RetrievedChunk, bank_keys: frozenset[str]) -> bool:
    """True if chunk metadata matches any slug in ``bank_keys`` (bank_key or bank_name)."""

    bk = (c.chunk.bank_key or "").strip().lower()
    bn = (c.chunk.bank_name or "").strip().lower()
    for raw in bank_keys:
        want = raw.strip().lower()
        if not want:
            continue
        if want == bk or want == bn:
            return True
    return False


def filter_chunks_to_bank_keys(
    chunks: list[RetrievedChunk],
    bank_keys: frozenset[str] | None,
) -> list[RetrievedChunk]:
    """
    Hard post-filter when the query scope names specific banks (OR semantics).

    Keeps the vector-store contract honest if a retriever returns stray rows; drops chunks
    with unknown bank metadata when ``bank_keys`` is non-empty.
    """

    if not bank_keys:
        return list(chunks)
    return [c for c in chunks if chunk_matches_bank_keys(c, bank_keys)]


def _chunk_identity(c: RetrievedChunk) -> str:
    cid = (c.chunk.chunk_id or "").strip()
    return cid if cid else f"__noid:{id(c)}"


def _diversify_coverage(
    ordered: list[RetrievedChunk],
    *,
    pool_limit: int,
    max_per_bank: int,
    max_per_url: int | None,
) -> list[RetrievedChunk]:
    """
    Prefer coverage across banks (and optionally canonical source URLs) for general RAG.

    First pass respects caps; second pass fills remaining slots with any unseen chunk_ids,
    still skipping duplicates but relaxing bank/url caps (mirrors prior per-bank behavior).
    """

    if pool_limit <= 0 or not ordered:
        return []
    bank_counts: dict[str, int] = {}
    url_counts: dict[str, int] = {}
    out: list[RetrievedChunk] = []
    id_seen: set[str] = set()

    def url_key(c: RetrievedChunk) -> str:
        return normalize_http_url(c.chunk.source_url or "") or "_nourl"

    for c in ordered:
        bk = _bank_key_chunk(c)
        uk = url_key(c)
        if bank_counts.get(bk, 0) >= max_per_bank:
            continue
        if max_per_url is not None and url_counts.get(uk, 0) >= max_per_url:
            continue
        ident = _chunk_identity(c)
        if ident in id_seen:
            continue
        id_seen.add(ident)
        out.append(c)
        bank_counts[bk] = bank_counts.get(bk, 0) + 1
        url_counts[uk] = url_counts.get(uk, 0) + 1
        if len(out) >= pool_limit:
            return out

    for c in ordered:
        ident = _chunk_identity(c)
        if ident in id_seen:
            continue
        uk = url_key(c)
        if max_per_url is not None and url_counts.get(uk, 0) >= max_per_url:
            continue
        id_seen.add(ident)
        out.append(c)
        url_counts[uk] = url_counts.get(uk, 0) + 1
        if len(out) >= pool_limit:
            break
    return out


def rerank_and_select(
    chunks: list[RetrievedChunk],
    query: str,
    top_k: int,
    *,
    diversify_banks: bool = False,
    max_per_bank: int = 2,
    max_per_source_url: int | None = 2,
) -> list[RetrievedChunk]:
    """
    Re-score FAISS hits using lightweight lexical overlap + anti-nav heuristics.

    Keeps retrieval fast (no extra model calls) while demoting menu-ish chunks.
    """

    if not chunks or top_k <= 0:
        return []
    scored: list[tuple[float, RetrievedChunk]] = []
    for c in chunks:
        body = c.chunk.cleaned_text
        sub = content_substance_score(body)
        ov = query_term_overlap(query, body)
        base = float(c.score)
        # Slightly higher lexical weight helps topical Armenian queries; dense+BM25 scores stay in `base`.
        combined = base * (0.48 + 0.40 * sub) + 0.20 * ov
        scored.append((combined, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = [c for _, c in scored]

    scan_pool = ordered
    if diversify_banks and len(ordered) > 1:
        pool_limit = min(len(ordered), max(top_k * 6, 48))
        per_url = max_per_source_url if max_per_source_url and max_per_source_url > 0 else None
        scan_pool = _diversify_coverage(
            ordered,
            pool_limit=pool_limit,
            max_per_bank=max_per_bank,
            max_per_url=per_url,
        )

    filtered: list[RetrievedChunk] = []
    for c in scan_pool:
        body = c.chunk.cleaned_text
        sub = content_substance_score(body)
        if sub < 0.11 and len(body) < 55:
            continue
        filtered.append(c)
        if len(filtered) >= top_k:
            break

    if len(filtered) >= min(2, top_k):
        return filtered
    return ordered[:top_k]
