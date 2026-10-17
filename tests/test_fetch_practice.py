"""End-to-end test of the practice pipeline: Jolpica tables + OpenF1 data -> saved laps."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from f1predict import config
from f1predict.data.fetch import fetch_practice, practice_seasons
from f1predict.data.http import FetchStats
from f1predict.data.tables import build_tables, save_season_tables

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


class FakeOpenF1:
    """Stands in for OpenF1Client: answers from the saved Singapore FP1 responses."""

    def __init__(self, openf1_fixture) -> None:
        self.load = openf1_fixture
        self.stats = FetchStats()

    def __enter__(self) -> FakeOpenF1:
        """Support ``with client:`` like the real client."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Nothing to close."""

    def get_json(self, path: str, params: dict[str, Any] | None = None, **_: Any) -> Any:
        """Return saved records for ``sessions`` and Singapore FP1; empty lists otherwise."""
        self.stats.downloads += 1
        if path == "sessions":
            return self.load("sessions_2026.json")
        if (params or {}).get("session_key") == 11378:
            return self.load(f"{path}_11378.json")
        return []


@pytest.fixture
def interim(tmp_path, monkeypatch, jolpica_races):
    """A temp interim folder holding the 2026 Jolpica tables built from the fixtures."""
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path)
    tables = build_tables(
        schedule=jolpica_races("schedule_2026.json"),
        results=jolpica_races("results_2026_17.json"),
        qualifying=jolpica_races("qualifying_2026_17.json"),
        sprint=jolpica_races("sprint_2026_17.json"),
    )
    save_season_tables(tables, 2026, tmp_path)
    return tmp_path


def test_practice_seasons_start_in_2023() -> None:
    """OpenF1 has nothing before 2023, so earlier seasons are skipped."""
    assert list(practice_seasons(2018, 2026)) == [2023, 2024, 2025, 2026]
    assert not practice_seasons(2018, 2022)
    assert fetch_practice(2018, 2022) == []


def test_practice_pipeline_saves_mapped_laps(interim, openf1_fixture) -> None:
    """Eight finished sessions; Singapore FP1's laps are saved with Jolpica driver ids."""
    summaries = fetch_practice(2026, 2026, now=NOW, client=FakeOpenF1(openf1_fixture))
    summary = summaries[0]
    assert (summary.season, summary.sessions, summary.laps) == (2026, 8, 62)
    assert summary.drivers == 3
    assert summary.practice_only == 0
    saved = pd.read_parquet(interim / "practice" / "2026.parquet")
    assert set(saved["driver_id"]) == {"norris", "stroll", "hulkenberg"}
    assert not (interim / "practice_only" / "2026.parquet").exists()
    assert sum("no timed laps" in w for w in summary.warnings) == 7
