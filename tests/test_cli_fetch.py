"""End-to-end tests for ``python -m f1predict fetch`` with faked HTTP and temporary folders."""

from __future__ import annotations

import pandas as pd
import pytest
import responses
from responses import matchers
from typer.testing import CliRunner

from f1predict import config
from f1predict.__main__ import app

BASE = "https://api.jolpi.ca/ergast/f1/"
FIRST_PAGE = [matchers.query_param_matcher({"limit": "100", "offset": "0"})]
runner = CliRunner()


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Point the cache, tables and logs at a temp folder, and switch off rate-limit waits."""
    monkeypatch.setattr(config, "JOLPICA_CACHE_DIR", tmp_path / "raw" / "jolpica")
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "interim")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "JOLPICA_MIN_INTERVAL_S", 0.0)
    return tmp_path


def test_fetch_rejects_seasons_outside_the_data_range() -> None:
    """Seasons before 2018 are refused before anything is downloaded."""
    result = runner.invoke(app, ["fetch", "--from", "2017"])
    assert result.exit_code == 2


def test_fetch_rejects_from_after_to() -> None:
    """A reversed range is a usage error."""
    result = runner.invoke(app, ["fetch", "--from", "2026", "--to", "2025"])
    assert result.exit_code == 2
    assert "later than" in result.output


def test_fetch_downloads_once_then_uses_the_cache(isolated_dirs, jolpica_fixture) -> None:
    """First run downloads 3 responses and writes tables; the second run sends nothing."""
    with responses.RequestsMock() as rsps:
        for path, name in (
            ("2018.json", "schedule_2018.json"),
            ("2018/results.json", "results_2018_1.json"),
            ("2018/qualifying.json", "qualifying_2018_1.json"),
        ):
            rsps.add(responses.GET, BASE + path, json=jolpica_fixture(name), match=FIRST_PAGE)

        first = runner.invoke(app, ["fetch", "--from", "2018", "--to", "2018"])
        assert first.exit_code == 0, first.output
        assert len(rsps.calls) == 3

        second = runner.invoke(app, ["fetch", "--from", "2018", "--to", "2018"])
        assert second.exit_code == 0, second.output
        assert len(rsps.calls) == 3

    results = pd.read_parquet(isolated_dirs / "interim" / "results" / "2018.parquet")
    assert len(results) == 20
    assert "Jolpica download" in second.output
    assert "start in 2023" in second.output
    assert (isolated_dirs / "logs" / "fetch.log").exists()


def test_no_practice_flag_skips_the_practice_step(isolated_dirs, jolpica_fixture) -> None:
    """``--no-practice`` downloads race data only and says nothing about practice."""
    with responses.RequestsMock() as rsps:
        for path, name in (
            ("2018.json", "schedule_2018.json"),
            ("2018/results.json", "results_2018_1.json"),
            ("2018/qualifying.json", "qualifying_2018_1.json"),
        ):
            rsps.add(responses.GET, BASE + path, json=jolpica_fixture(name), match=FIRST_PAGE)
        result = runner.invoke(app, ["fetch", "--from", "2018", "--to", "2018", "--no-practice"])
    assert result.exit_code == 0, result.output
    assert "OpenF1" not in result.output


def test_fetch_failure_exits_with_code_1_and_explains(isolated_dirs) -> None:
    """A hard API error stops the command with a readable message, not a traceback."""
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, BASE + "2018.json", json={}, status=404, match=FIRST_PAGE)
        result = runner.invoke(app, ["fetch", "--from", "2018", "--to", "2018"])
    assert result.exit_code == 1
    assert "Fetch failed" in result.output
