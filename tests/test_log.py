"""Tests for the shared logging setup."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from f1predict.log import LOGGER_NAME, setup_logging


@pytest.fixture(autouse=True)
def clean_logger() -> Iterator[None]:
    """Remove and close the handlers a test added, so log files are released on Windows."""
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def test_messages_reach_the_log_file(tmp_path) -> None:
    """INFO messages are written to the log file (and its folder is created); DEBUG is not."""
    log_file = tmp_path / "logs" / "run.log"
    logger = setup_logging(level="INFO", log_file=log_file)
    logger.info("lap times saved")
    logger.debug("hidden detail")
    text = log_file.read_text(encoding="utf-8")
    assert "lap times saved" in text
    assert "hidden detail" not in text


def test_setup_twice_does_not_stack_handlers(tmp_path) -> None:
    """Calling setup again replaces the handlers instead of adding duplicates."""
    setup_logging(log_file=tmp_path / "first.log")
    count = len(logging.getLogger(LOGGER_NAME).handlers)
    setup_logging(log_file=tmp_path / "second.log")
    assert len(logging.getLogger(LOGGER_NAME).handlers) == count
