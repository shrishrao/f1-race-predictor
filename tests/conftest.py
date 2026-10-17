"""Shared pytest fixtures.

Every test runs with network access disabled (``--disable-socket`` in ``pyproject.toml``),
so tests can only use saved sample data from ``tests/fixtures/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the folder of saved sample data used by tests."""
    return FIXTURES_DIR
