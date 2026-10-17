"""Tests for the Jolpica client: retries, pagination, caching and error handling.

HTTP is faked with the ``responses`` library, so these tests never touch the network. The
rate limiter itself is tested in ``test_http.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import requests
import responses
from responses import matchers

from f1predict.data.cache import JsonCache
from f1predict.data.http import FetchStats, RateLimiter
from f1predict.data.jolpica import JolpicaClient, JolpicaError, JolpicaHTTPError, merge_pages

BASE = "https://api.jolpi.ca/ergast/f1/"
T0 = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def page(races: list[dict[str, Any]], *, total: int, limit: int, offset: int) -> dict[str, Any]:
    """Build a response in Jolpica's MRData format."""
    return {
        "MRData": {
            "limit": str(limit),
            "offset": str(offset),
            "total": str(total),
            "RaceTable": {"Races": races},
        }
    }


def race(round_number: int, drivers: list[str]) -> dict[str, Any]:
    """Build one race with a ``Results`` row per driver id."""
    return {
        "season": "2025",
        "round": str(round_number),
        "Results": [{"Driver": {"driverId": d}} for d in drivers],
    }


def driver_ids(races: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """Summarize merged races as (round, [driver ids]) pairs."""
    return [(r["round"], [e["Driver"]["driverId"] for e in r["Results"]]) for r in races]


def add(
    rsps: responses.RequestsMock,
    path: str,
    payload: Any,
    *,
    offset: int = 0,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> None:
    """Register a fake response for one page of ``path``."""
    rsps.add(
        responses.GET,
        BASE + path,
        json=payload,
        status=status,
        headers=headers,
        match=[matchers.query_param_matcher({"limit": "100", "offset": str(offset)})],
    )


def make_client(tmp_path, **kwargs: Any) -> JolpicaClient:
    """A client with a temp cache, no rate-limit waits, and fake sleep/now unless overridden."""
    kwargs.setdefault("limiter", RateLimiter(0.0, 1_000_000, 3600.0, sleep=lambda seconds: None))
    kwargs.setdefault("sleep", lambda seconds: None)
    kwargs.setdefault("now", lambda: T0)
    return JolpicaClient(JsonCache(tmp_path / "cache"), **kwargs)


@pytest.fixture
def rsps():
    """Fake HTTP for the test; any request that wasn't registered fails."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_pages_follow_the_servers_limit_and_split_races_are_rejoined(tmp_path, rsps) -> None:
    """The server caps pages at 3 rows; round 2 is split across pages and gets stitched back."""
    first = page([race(1, ["a", "b"]), race(2, ["c"])], total=5, limit=3, offset=0)
    second = page([race(2, ["d"]), race(3, ["e"])], total=5, limit=3, offset=3)
    add(rsps, "2025/results.json", first)
    add(rsps, "2025/results.json", second, offset=3)
    races = make_client(tmp_path).get_races("2025/results.json", results_key="Results")
    assert driver_ids(races) == [("1", ["a", "b"]), ("2", ["c", "d"]), ("3", ["e"])]
    assert len(rsps.calls) == 2


def test_merge_without_results_key_keeps_one_entry_per_round() -> None:
    """Schedule pages have no results lists; races are simply de-duplicated by round."""
    pages = [
        page([race(1, [])], total=2, limit=1, offset=0),
        page([race(2, [])], total=2, limit=1, offset=1),
    ]
    assert [r["round"] for r in merge_pages(pages)] == ["1", "2"]


def test_zero_page_limit_is_an_error(tmp_path, rsps) -> None:
    """A page limit of 0 would loop forever, so it raises instead."""
    add(rsps, "2025.json", page([], total=5, limit=0, offset=0))
    with pytest.raises(JolpicaError, match="page limit"):
        make_client(tmp_path).get_races("2025.json")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_second_request_is_served_from_cache(tmp_path, rsps) -> None:
    """Asking twice sends one request; a new client reads the same files from disk."""
    add(rsps, "2025.json", page([race(1, [])], total=1, limit=100, offset=0))
    client = make_client(tmp_path)
    first = client.get_races("2025.json")
    assert client.get_races("2025.json") == first
    assert client.stats == FetchStats(downloads=1, cache_hits=1)
    assert make_client(tmp_path).get_races("2025.json") == first
    assert len(rsps.calls) == 1


def test_empty_responses_are_never_cached(tmp_path, rsps) -> None:
    """A session Jolpica hasn't published yet is asked for again next time."""
    add(rsps, "2026/18/qualifying.json", page([], total=0, limit=100, offset=0))
    client = make_client(tmp_path)
    assert client.get_races("2026/18/qualifying.json") == []
    assert client.get_races("2026/18/qualifying.json") == []
    assert len(rsps.calls) == 2
    assert list(tmp_path.rglob("*.json")) == []


def test_cached_copy_older_than_fresh_since_is_downloaded_again(tmp_path, rsps) -> None:
    """A copy downloaded before ``fresh_since`` is replaced; a newer one is reused."""
    path, params = "2026/17/results.json", {"limit": 100, "offset": 0}
    cache = JsonCache(tmp_path / "cache")
    old = page([race(17, ["old"])], total=1, limit=100, offset=0)
    cache.save(path, params, url="u", payload=old, fetched_at=T0)
    add(rsps, path, page([race(17, ["new"])], total=1, limit=100, offset=0))
    client = make_client(tmp_path, now=lambda: T0 + timedelta(hours=2))

    reused = client.get_races(path, results_key="Results", fresh_since=T0 - timedelta(hours=1))
    assert driver_ids(reused) == [("17", ["old"])]
    assert len(rsps.calls) == 0

    replaced = client.get_races(path, results_key="Results", fresh_since=T0 + timedelta(hours=1))
    assert driver_ids(replaced) == [("17", ["new"])]
    assert len(rsps.calls) == 1
    assert cache.load(path, params).fetched_at == T0 + timedelta(hours=2)


def test_refresh_ignores_the_cache(tmp_path, rsps) -> None:
    """``--refresh`` downloads again even when a cached copy exists."""
    cache = JsonCache(tmp_path / "cache")
    cached = page([], total=1, limit=100, offset=0)
    cache.save("2025.json", {"limit": 100, "offset": 0}, url="u", payload=cached, fetched_at=T0)
    add(rsps, "2025.json", page([race(1, [])], total=1, limit=100, offset=0))
    make_client(tmp_path, refresh=True).get_races("2025.json")
    assert len(rsps.calls) == 1


# ---------------------------------------------------------------------------
# Retries, timeouts and errors
# ---------------------------------------------------------------------------


def test_rate_limited_request_waits_for_retry_after_then_succeeds(tmp_path, rsps) -> None:
    """HTTP 429 with Retry-After: 7 waits at least 7 s, slows the limiter, then succeeds."""
    add(rsps, "2025.json", {"detail": "throttled"}, status=429, headers={"Retry-After": "7"})
    add(rsps, "2025.json", page([race(1, [])], total=1, limit=100, offset=0))
    sleeps: list[float] = []
    client = make_client(tmp_path, sleep=sleeps.append)
    races = client.get_races("2025.json")
    assert [r["round"] for r in races] == ["1"]
    assert len(rsps.calls) == 2
    assert len(sleeps) == 1
    assert sleeps[0] >= 7
    assert client.stats == FetchStats(downloads=1, retries=1)
    assert client.limiter.min_interval == 0.5


def test_server_errors_give_up_after_max_attempts(tmp_path, rsps) -> None:
    """HTTP 503 every time: 3 attempts, 2 waits, then the error is raised."""
    add(rsps, "2025.json", {}, status=503)
    sleeps: list[float] = []
    with pytest.raises(JolpicaHTTPError) as info:
        make_client(tmp_path, max_attempts=3, sleep=sleeps.append).get_races("2025.json")
    assert info.value.status == 503
    assert len(rsps.calls) == 3
    assert len(sleeps) == 2


def test_client_errors_are_not_retried(tmp_path, rsps) -> None:
    """HTTP 404 means the request is wrong; retrying cannot help, so it fails at once."""
    add(rsps, "2025.json", {}, status=404)
    sleeps: list[float] = []
    with pytest.raises(JolpicaHTTPError):
        make_client(tmp_path, sleep=sleeps.append).get_races("2025.json")
    assert len(rsps.calls) == 1
    assert sleeps == []


def test_dropped_connections_are_retried_then_reported(tmp_path, rsps) -> None:
    """A connection error is retried; if it keeps failing it becomes a clear JolpicaError."""
    rsps.add(
        responses.GET,
        BASE + "2025.json",
        body=requests.ConnectionError("connection reset"),
        match=[matchers.query_param_matcher({"limit": "100", "offset": "0"})],
    )
    with pytest.raises(JolpicaError, match="Could not reach Jolpica"):
        make_client(tmp_path, max_attempts=2).get_races("2025.json")
    assert len(rsps.calls) == 2


def test_invalid_json_is_an_error(tmp_path, rsps) -> None:
    """An HTML error page instead of JSON is reported, not cached."""
    rsps.add(
        responses.GET,
        BASE + "2025.json",
        body="<html>maintenance</html>",
        content_type="text/html",
        match=[matchers.query_param_matcher({"limit": "100", "offset": "0"})],
    )
    with pytest.raises(JolpicaError, match="invalid JSON"):
        make_client(tmp_path).get_races("2025.json")
    assert list(tmp_path.rglob("*.json")) == []


def test_requests_identify_the_client_and_always_use_timeouts(tmp_path, rsps) -> None:
    """Every request sends our User-Agent and a (connect, read) timeout."""
    add(rsps, "2025.json", page([race(1, [])], total=1, limit=100, offset=0))
    make_client(tmp_path).get_races("2025.json")
    request = rsps.calls[0].request
    assert request.headers["User-Agent"].startswith("f1predict/")
    assert request.req_kwargs["timeout"] == (5.0, 30.0)
