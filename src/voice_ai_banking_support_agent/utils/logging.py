from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Configure root logging for the whole project.

    Args:
        log_level: Standard logging level string, e.g. "INFO", "DEBUG".
        log_file: Optional file path for rotating logs.
    """

    root = logging.getLogger()
    desired_level = log_level.upper()
    root.setLevel(desired_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    has_console = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
    else:
        for h in root.handlers:
            h.setLevel(desired_level)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_path = str(log_file.resolve())
        has_file = any(
            isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == file_path
            for h in root.handlers
        )
        if not has_file:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(desired_level)
            root.addHandler(file_handler)

