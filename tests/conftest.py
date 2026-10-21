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


def _close_log_files() -> None:
    """Remove and close the handlers ``setup_logging`` added."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture(autouse=True)
def close_log_handlers() -> Iterator[None]:
    """After each test, remove and close handlers added by setup_logging.

    Open log files cannot be deleted on Windows, so leaving them open would break the
    cleanup of pytest's temporary folders.
    """
    yield
    _close_log_files()


# ---------------------------------------------------------------------------
# A published workspace (publish and app tests)
# ---------------------------------------------------------------------------

WORKSPACE_FOLDERS = {
    "INTERIM_DIR": "interim",
    "PROCESSED_DIR": "processed",
    "MODELS_DIR": "models",
    "LOG_DIR": "logs",
    "PUBLISHED_DIR": "published",
}


def point_config_at(patch: pytest.MonkeyPatch, root: Path) -> None:
    """Point every data and model folder in ``config`` at ``root``'s subfolders."""
    from f1predict import config

    for name, folder in WORKSPACE_FOLDERS.items():
        patch.setattr(config, name, root / folder)


@pytest.fixture(scope="session")
def published_workspace(tmp_path_factory) -> Path:
    """Synthetic data, trained models, 2024 backtests and published files, built once.

    2024 round 5 has been qualified but not run, so the published files hold completed
    races and an upcoming one. Everything is made by the real commands and ``publish``;
    ``config`` is put back afterwards, and tests that need it use ``in_workspace``.
    """
    from synthetic import make_tables
    from typer.testing import CliRunner

    from f1predict.__main__ import app
    from f1predict.data.tables import save_season_tables
    from f1predict.features.build import build_features, save_features
    from f1predict.publish import publish

    root = tmp_path_factory.mktemp("workspace")
    full = make_tables()
    upcoming = (full.results["season"] == 2024) & (full.results["round"] == 5)
    tables = type(full)(
        full.schedule, full.results[~upcoming], full.qualifying, full.sprint, full.practice
    )
    with pytest.MonkeyPatch.context() as patch:
        point_config_at(patch, root)
        for season in (2023, 2024):
            frames = {
                name: frame[frame["season"] == season] for name, frame in vars(tables).items()
            }
            save_season_tables(frames, season)
        save_features(build_features(tables))
        runner = CliRunner()
        for args in (
            ["train"],
            ["backtest", "--season", "2024"],
            ["backtest", "--model", "ranker", "--season", "2024"],
        ):
            result = runner.invoke(app, args)
            assert result.exit_code == 0, result.output
        publish(seasons=(2024,))
        _close_log_files()
    return root


@pytest.fixture
def in_workspace(published_workspace, monkeypatch) -> Path:
    """``config`` pointed at the published workspace for one test."""
    point_config_at(monkeypatch, published_workspace)
    return published_workspace
