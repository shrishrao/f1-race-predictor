"""Tests for which requests a season download makes, and how fresh the cache must be."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from f1predict.data.jolpica import download_season

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
SINGAPORE_RACE = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
SEPANG_RACE = datetime(2026, 10, 4, 7, 0, tzinfo=UTC)


class FakeSource:
    """Records every request; returns the saved schedule, or one empty race per round."""

    def __init__(self, schedule: list[dict[str, Any]], *, published: bool = True) -> None:
        self.schedule = schedule
        self.published = published
        self.calls: list[tuple[str, datetime | None]] = []

    def get_races(
        self,
        path: str,
        *,
        results_key: str | None = None,
        fresh_since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Record the request and return canned races."""
        self.calls.append((path, fresh_since))
        if "/" not in path:
            return self.schedule
        parts = path.split("/")
        if len(parts) == 3 and self.published:
            return [{"season": parts[0], "round": parts[1], results_key: []}]
        return []


def test_finished_season_uses_season_wide_requests_cached_forever() -> None:
    """A past season asks for whole-season endpoints once, with no freshness limit."""
    source = FakeSource([])
    download_season(source, 2025, now=NOW)
    assert source.calls == [
        ("2025.json", None),
        ("2025/results.json", None),
        ("2025/qualifying.json", None),
        ("2025/sprint.json", None),
    ]


def test_no_sprint_request_before_sprints_existed() -> None:
    """Seasons before 2021 had no sprints, so that endpoint is skipped."""
    source = FakeSource([])
    download_season(source, 2019, now=NOW)
    assert [path for path, _ in source.calls] == [
        "2019.json",
        "2019/results.json",
        "2019/qualifying.json",
    ]


def test_current_season_only_asks_for_sessions_that_have_started(jolpica_races) -> None:
    """Round 18 is in the future (nothing asked); sprint requests only on sprint weekends."""
    source = FakeSource(jolpica_races("schedule_2026.json"))
    download_season(source, 2026, now=NOW)
    assert [path for path, _ in source.calls] == [
        "2026.json",
        "2026/1/results.json",
        "2026/1/qualifying.json",
        "2026/2/results.json",
        "2026/2/qualifying.json",
        "2026/2/sprint.json",
        "2026/16/results.json",
        "2026/16/qualifying.json",
        "2026/17/results.json",
        "2026/17/qualifying.json",
        "2026/17/sprint.json",
    ]


def test_current_season_freshness_rules(jolpica_races) -> None:
    """The calendar is re-checked daily; a settled race needs a copy from after it settled."""
    source = FakeSource(jolpica_races("schedule_2026.json"))
    download_season(source, 2026, now=NOW)
    fresh = dict(source.calls)
    assert fresh["2026.json"] == NOW - timedelta(hours=24)
    assert fresh["2026/17/results.json"] == SINGAPORE_RACE + timedelta(hours=72)


def test_recent_session_is_rechecked_hourly_until_it_settles(jolpica_races) -> None:
    """Ten hours after Singapore, its results may still change; Sepang has already settled."""
    now = SINGAPORE_RACE + timedelta(hours=10)
    source = FakeSource(jolpica_races("schedule_2026.json"))
    download_season(source, 2026, now=now)
    fresh = dict(source.calls)
    assert fresh["2026/17/results.json"] == now - timedelta(hours=1)
    assert fresh["2026/16/results.json"] == SEPANG_RACE + timedelta(hours=72)


def test_unpublished_sessions_are_left_out(jolpica_races) -> None:
    """If Jolpica has nothing yet, the season data simply has no rows for those sessions."""
    source = FakeSource(jolpica_races("schedule_2026.json"), published=False)
    data = download_season(source, 2026, now=NOW)
    assert len(data.schedule) == 5
    assert data.results == []
    assert data.qualifying == []
    assert data.sprint == []
