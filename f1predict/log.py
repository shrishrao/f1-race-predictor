"""Logging shared by every command: Rich console output plus an optional log file."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

LOGGER_NAME = "f1predict"
FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure and return the ``f1predict`` logger.

    Console output goes through Rich. If ``log_file`` is given, the same messages are also
    appended to that file (its folder is created if needed), which matters for downloads
    that run for hours. Calling this again replaces the previous handlers instead of
    stacking new ones.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(level.upper())

    console_handler = RichHandler(rich_tracebacks=True, show_path=False, log_time_format="[%X]")
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        logger.addHandler(file_handler)

    return logger
