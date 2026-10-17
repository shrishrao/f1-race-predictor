"""Shared pytest fixtures.

Every test runs with network access disabled (``--disable-socket`` in ``pyproject.toml``),
so tests can only use saved sample data from ``tests/fixtures/``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from f1predict.log import LOGGER_NAME

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the folder of saved sample data used by tests."""
    return FIXTURES_DIR


@pytest.fixture
def jolpica_fixture() -> Callable[[str], dict[str, Any]]:
    """Load one saved Jolpica response by file name (a fresh copy on every call)."""

    def load(name: str) -> dict[str, Any]:
        """Read ``tests/fixtures/jolpica/<name>`` as JSON."""
        return json.loads((FIXTURES_DIR / "jolpica" / name).read_text(encoding="utf-8"))

    return load


@pytest.fixture
def jolpica_races(jolpica_fixture) -> Callable[[str], list[dict[str, Any]]]:
    """Load just the list of races from one saved Jolpica response."""

    def load(name: str) -> list[dict[str, Any]]:
        """Return ``MRData.RaceTable.Races`` from the saved response."""
        return jolpica_fixture(name)["MRData"]["RaceTable"]["Races"]

    return load


@pytest.fixture
def openf1_fixture() -> Callable[[str], Any]:
    """Load one saved OpenF1 response by file name (a fresh copy on every call)."""

    def load(name: str) -> Any:
        """Read ``tests/fixtures/openf1/<name>`` as JSON."""
        return json.loads((FIXTURES_DIR / "openf1" / name).read_text(encoding="utf-8"))

    return load


@pytest.fixture(autouse=True)
def close_log_handlers() -> Iterator[None]:
    """After each test, remove and close handlers added by setup_logging.

    Open log files cannot be deleted on Windows, so leaving them open would break the
    cleanup of pytest's temporary folders.
    """
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
