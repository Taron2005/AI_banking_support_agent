from __future__ import annotations

import hashlib
import re
from typing import Any


def slugify(text: str, *, max_length: int = 80) -> str:
    """Convert text to a filesystem-friendly slug."""

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip("-")


def stable_id(*parts: Any, length: int = 16) -> str:
    """Generate a stable short id from multiple parts."""

    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:length]

