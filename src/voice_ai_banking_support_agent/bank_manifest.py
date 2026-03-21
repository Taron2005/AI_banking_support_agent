from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import BanksManifest


def _validate_url(url: str) -> None:
    """Validate that a URL looks like an HTTP(S) link."""

    url = url.strip()
    if not url:
        raise ValueError("Empty URL is not allowed.")
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        raise ValueError(f"Invalid URL (must start with http/https): {url}")


def load_banks_manifest(manifest_path: Path) -> BanksManifest:
    """
    Load and validate `manifests/banks.yaml`.

    This function enforces that:
    - URL lists are present for allowed topics,
    - each URL resembles an HTTP(S) URL.
    """

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Manifest YAML root must be a mapping/object.")

    schema_version = str(raw.get("schema_version", ""))
    if schema_version != "1":
        raise ValueError(f"Unsupported manifest schema_version={schema_version!r}. Expected '1'.")

    # Pre-validate URLs and bank keys to fail fast with clearer messages.
    banks = raw.get("banks", [])
    seen_bank_keys: set[str] = set()
    if isinstance(banks, list):
        for bank in banks:
            if not isinstance(bank, dict):
                continue
            bank_key = str(bank.get("bank_key", "")).strip().lower()
            if not bank_key:
                raise ValueError("Manifest bank entry missing non-empty bank_key.")
            if bank_key in seen_bank_keys:
                raise ValueError(f"Duplicate bank_key in manifest: {bank_key}")
            seen_bank_keys.add(bank_key)
            for topic_key in ["credits", "deposits", "branches"]:
                topic_obj = bank.get(topic_key, {})
                if not isinstance(topic_obj, dict):
                    raise ValueError(f"Manifest topic entry must be object: bank={bank_key} topic={topic_key}")
                urls = topic_obj.get("urls", [])
                if not isinstance(urls, list):
                    raise ValueError(f"Manifest urls must be list: bank={bank_key} topic={topic_key}")
                if not urls:
                    raise ValueError(f"Manifest urls list cannot be empty: bank={bank_key} topic={topic_key}")
                deduped_urls: list[str] = []
                seen_urls: set[str] = set()
                for u in urls:
                    if isinstance(u, str):
                        _validate_url(u)
                        u_norm = u.strip()
                        if u_norm not in seen_urls:
                            seen_urls.add(u_norm)
                            deduped_urls.append(u_norm)
                topic_obj["urls"] = deduped_urls

    try:
        return BanksManifest.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Manifest validation error: {e}") from e


def manifest_summary(manifest: BanksManifest) -> str:
    """Create a compact human-readable manifest summary (for logging/CLI)."""

    lines: list[str] = []
    lines.append(f"Manifest schema_version={manifest.schema_version}")
    for b in manifest.banks:
        lines.append(
            f"- {b.bank_key} ({b.bank_name}): credits={len(b.credits.urls)}, "
            f"deposits={len(b.deposits.urls)}, branches={len(b.branches.urls)}"
        )
    return "\n".join(lines)

