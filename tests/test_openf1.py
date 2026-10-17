"""Tests for the OpenF1 client and for matching OpenF1 practice sessions to Jolpica rounds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import responses
from responses import matchers

from f1predict.data.cache import JsonCache
from f1predict.data.http import FetchStats, RateLimiter
from f1predict.data.openf1 import (
    OpenF1Client,
    OpenF1HTTPError,
    PracticeAlignmentError,
    download_practice_season,
    match_practice_sessions,
)
from f1predict.data.tables import schedule_table

BASE = "https://api.openf1.org/v1/"
NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
SINGAPORE_FP1_END = datetime(2026, 10, 9, 9, 30, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client(tmp_path, **kwargs: Any) -> OpenF1Client:
    """A client with a temp cache, no rate-limit waits, and a fixed clock."""
    kwargs.setdefault("limiter", RateLimiter(0.0, 1_000_000, 60.0, sleep=lambda seconds: None))
    kwargs.setdefault("sleep", lambda seconds: None)
    kwargs.setdefault("now", lambda: NOW)
    return OpenF1Client(JsonCache(tmp_path / "cache"), **kwargs)


@pytest.fixture
def rsps():
    """Fake HTTP for the test; any request that wasn't registered fails."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock


@pytest.fixture
def schedule(jolpica_races):
    """Jolpica's 2026 schedule for rounds 1, 2, 16, 17 and 18."""
    return schedule_table(jolpica_races("schedule_2026.json"))


class FakeClient:
    """Records requests and answers from the saved Singapore FP1 responses."""

    def __init__(self, openf1_fixture) -> None:
        self.load = openf1_fixture
        self.calls: list[tuple[str, dict[str, Any], datetime | None, bool]] = []
        self.stats = FetchStats()

    def get_json(self, path, params=None, *, fresh_since=None, cache_empty=False):
        """Return saved records for ``sessions`` and Singapore FP1; empty lists otherwise."""
        self.calls.append((path, dict(params or {}), fresh_since, cache_empty))
        if path == "sessions":
            return self.load("sessions_2026.json")
        if params.get("session_key") == 11378:
            return self.load(f"{path}_11378.json")
        return []


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def test_no_results_404_means_an_empty_list(tmp_path, rsps) -> None:
    """OpenF1 answers 404 "No results found." for an empty query; that's data, not an error."""
    rsps.add(
        responses.GET,
        BASE + "laps",
        json={"detail": "No results found."},
        status=404,
        match=[matchers.query_param_matcher({"session_key": "1"})],
    )
    client = make_client(tmp_path)
    assert client.get_json("laps", {"session_key": 1}) == []
    assert client.get_json("laps", {"session_key": 1}, cache_empty=True) == []
    assert client.get_json("laps", {"session_key": 1}, cache_empty=True) == []
    assert len(rsps.calls) == 2
    assert client.stats == FetchStats(downloads=2, cache_hits=1)


def test_other_404s_are_still_errors(tmp_path, rsps) -> None:
    """A 404 without the "No results" message means a wrong URL, and is raised."""
    rsps.add(responses.GET, BASE + "lapz", json={"detail": "Not Found"}, status=404)
    with pytest.raises(OpenF1HTTPError):
        make_client(tmp_path).get_json("lapz")


def test_non_list_response_is_rejected(tmp_path, rsps) -> None:
    """OpenF1 always returns a list; anything else is reported."""
    rsps.add(responses.GET, BASE + "laps", json={"unexpected": True})
    with pytest.raises(Exception, match="Unexpected response shape"):
        make_client(tmp_path).get_json("laps")


# ---------------------------------------------------------------------------
# Matching sessions to rounds
# ---------------------------------------------------------------------------


def test_sessions_match_rounds_by_number_and_start_time(openf1_fixture, schedule) -> None:
    """Normal weekends get FP1-FP3; sprint weekends (rounds 2 and 17) only FP1."""
    matched = match_practice_sessions(openf1_fixture("sessions_2026.json"), schedule, 2026)
    assert [(m.round, m.session) for m in matched] == [
        (1, "FP1"),
        (1, "FP2"),
        (1, "FP3"),
        (2, "FP1"),
        (16, "FP1"),
        (16, "FP2"),
        (16, "FP3"),
        (17, "FP1"),
        (18, "FP1"),
        (18, "FP2"),
        (18, "FP3"),
    ]
    singapore = next(m for m in matched if m.round == 17)
    assert singapore.session_key == 11378
    assert singapore.end == SINGAPORE_FP1_END


def test_testing_and_cancelled_sessions_are_skipped(openf1_fixture, schedule) -> None:
    """Pre-season testing days and cancelled sessions never reach the matcher."""
    sessions = openf1_fixture("sessions_2026.json")
    testing = dict(sessions[0], session_name="Day 1")
    cancelled = dict(sessions[1], is_cancelled=True)
    matched = match_practice_sessions([testing, cancelled, *sessions[2:]], schedule, 2026)
    keys = {m.session_key for m in matched}
    assert len(matched) == 9
    assert sessions[0]["session_key"] not in keys
    assert sessions[1]["session_key"] not in keys


def test_a_session_matching_no_round_fails_loudly(openf1_fixture, schedule) -> None:
    """A practice session that lines up with no Jolpica round stops the fetch."""
    session = dict(openf1_fixture("sessions_2026.json")[0], date_start="2026-07-01T12:00:00+00:00")
    with pytest.raises(PracticeAlignmentError, match="matches 0 Jolpica rounds"):
        match_practice_sessions([session], schedule, 2026)


def test_two_sessions_claiming_the_same_slot_fail_loudly(openf1_fixture, schedule) -> None:
    """Two OpenF1 sessions for the same round's FP1 can't both be right."""
    first = openf1_fixture("sessions_2026.json")[0]
    duplicate = dict(first, session_key=999)
    with pytest.raises(PracticeAlignmentError, match="several OpenF1 sessions"):
        match_practice_sessions([first, duplicate], schedule, 2026)


# ---------------------------------------------------------------------------
# Downloading a season
# ---------------------------------------------------------------------------


def test_season_download_asks_four_endpoints_per_finished_session(openf1_fixture, schedule) -> None:
    """Round 18 hasn't happened yet; every finished session costs exactly four requests."""
    client = FakeClient(openf1_fixture)
    sessions = download_practice_season(client, 2026, schedule, now=NOW)
    assert len(sessions) == 8
    session_calls = [call for call in client.calls if call[0] != "sessions"]
    assert len(session_calls) == 8 * 4
    endpoints = {call[0] for call in session_calls}
    assert endpoints == {"drivers", "laps", "stints", "race_control"}
    singapore = next(s for s in sessions if s.session.round == 17)
    assert len(singapore.laps) == 66


def test_settled_sessions_cache_even_empty_answers(openf1_fixture, schedule) -> None:
    """Sessions more than 72 hours old are final, so even an empty answer is cached."""
    client = FakeClient(openf1_fixture)
    now = SINGAPORE_FP1_END + timedelta(hours=10)
    download_practice_season(client, 2026, schedule, now=now)
    by_key = {(call[0], call[1].get("session_key")): call for call in client.calls}
    recent = by_key[("laps", 11378)]
    assert recent[2] == now - timedelta(hours=1)
    assert recent[3] is False
    older = [call for call in client.calls if call[0] == "laps" and call[1]["session_key"] != 11378]
    assert all(call[3] is True for call in older)


def test_a_day_long_disagreement_still_matches_with_a_warning(
    openf1_fixture, schedule, caplog
) -> None:
    """Jolpica dates the 2023 Las Vegas sessions a day early; such sessions still match."""
    first = openf1_fixture("sessions_2026.json")[0]
    later = datetime.fromisoformat(first["date_start"]) + timedelta(hours=24)
    shifted = dict(first, date_start=later.isoformat())
    with caplog.at_level("WARNING"):
        matched = match_practice_sessions([shifted], schedule, 2026)
    assert [(m.round, m.session) for m in matched] == [(1, "FP1")]
    assert "24.0 hours away" in caplog.text
