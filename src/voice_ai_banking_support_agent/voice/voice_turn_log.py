"""Structured per-turn logging for the LiveKit voice pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VoiceTurnLog:
    """One push-to-talk turn: state transitions and timings for debugging."""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _t0: float = field(default_factory=time.perf_counter)

    def event(self, stage: str, **extra: Any) -> None:
        elapsed_ms = round((time.perf_counter() - self._t0) * 1000, 1)
        if extra:
            logger.info(
                "voice_turn id=%s stage=%s elapsed_ms=%s extra=%s",
                self.turn_id,
                stage,
                elapsed_ms,
                extra,
            )
        else:
            logger.info("voice_turn id=%s stage=%s elapsed_ms=%s", self.turn_id, stage, elapsed_ms)

    def fail(self, stage: str, exc: BaseException) -> None:
        self.event(
            f"{stage}_failed",
            exc_type=type(exc).__name__,
            exc_msg=str(exc)[:500],
        )
