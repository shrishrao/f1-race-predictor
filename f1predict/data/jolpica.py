"""Jolpica F1 API client: rate limiting, retries, timeouts, pagination, and caching.

Every request passes through three layers:

1. The disk cache. If a fresh enough copy is on disk, nothing is sent at all.
2. The rate limiter. One request every 0.5 s at most (and 450 per hour), slowing down
   automatically whenever Jolpica answers HTTP 429 "too many requests".
3. Retries. Timeouts, dropped connections, HTTP 429 and HTTP 5xx are retried with
   exponential backoff; any other error fails straight away.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, Self

import requests
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from f1predict import __version__, config
from f1predict.data.cache import JsonCache
from f1predict.data.tables import session_starts

logger = logging.getLogger(__name__)

USER_AGENT = f"f1predict/{__version__} (+https://github.com/shrishrao/f1-race-predictor)"
MAX_RETRY_AFTER_S = 300.0
MIN_BACKOFF_INTERVAL_S = 0.5
RESULTS_KEYS = {
    "results": "Results",
    "qualifying": "QualifyingResults",
    "sprint": "SprintResults",
}
SESSION_OF_ENDPOINT = {"results": "race", "qualifying": "quali", "sprint": "sprint"}

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class JolpicaError(RuntimeError):
    """The Jolpica API could not be reached or returned something unusable."""


class JolpicaHTTPError(JolpicaError):
    """Jolpica answered with an HTTP error status."""

    def __init__(self, status: int, url: str, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status} from {url}")
        self.status = status
        self.url = url
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Space out calls, and widen the spacing when the server says "too many requests".

    Two fixed rules: calls are at least ``min_interval`` seconds apart, and at most
    ``max_calls`` happen per ``period`` (tracked as a sliding window of recent call times).
    On top of that the gap adapts. ``slow_down()`` doubles it after an HTTP 429 (up to
    ``max_interval``), and every ``recover_after`` successful calls in a row shrink it by a
    quarter, back toward the starting value. Jolpica's real limits are stricter than its
    docs and are expected to drop further, so the limiter finds a safe speed by itself.

    ``clock`` and ``sleep`` can be swapped for fakes in tests, so tests never really wait.
    """

    def __init__(
        self,
        min_interval: float,
        max_calls: int,
        period: float,
        *,
        max_interval: float = 8.0,
        recover_after: int = 20,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.base_interval = min_interval
        self.min_interval = min_interval
        self.max_interval = max(max_interval, min_interval)
        self.recover_after = recover_after
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._successes = 0

    def slow_down(self) -> None:
        """Double the gap between calls (at least 0.5 s, at most ``max_interval``)."""
        self.min_interval = min(
            max(self.min_interval * 2, MIN_BACKOFF_INTERVAL_S), self.max_interval
        )
        self._successes = 0

    def record_success(self) -> None:
        """Count a successful call; after enough in a row, shrink the gap by a quarter."""
        self._successes += 1
        if self._successes >= self.recover_after and self.min_interval > self.base_interval:
            self.min_interval = max(self.base_interval, self.min_interval * 0.75)
            self._successes = 0

    def acquire(self) -> None:
        """Wait until one more call is allowed, then record it."""
        now = self._clock()
        self._forget_up_to(now - self.period)
        if len(self._calls) >= self.max_calls:
            self._sleep(self._calls[0] + self.period - now)
            now = self._clock()
            self._forget_up_to(now - self.period)
        if self._calls:
            gap = self._calls[-1] + self.min_interval - now
            if gap > 0:
                self._sleep(gap)
                now = self._clock()
        self._calls.append(now)

    def _forget_up_to(self, cutoff: float) -> None:
        """Drop remembered calls made at or before ``cutoff``; they no longer count."""
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

_backoff = wait_exponential_jitter(initial=1, max=60)


def _is_retryable(exc: BaseException) -> bool:
    """Retry timeouts, dropped connections, HTTP 429 (slow down) and HTTP 5xx (server trouble)."""
    if isinstance(exc, requests.Timeout | requests.ConnectionError):
        return True
    return isinstance(exc, JolpicaHTTPError) and (exc.status == 429 or exc.status >= 500)


def _wait_seconds(state: RetryCallState) -> float:
    """Exponential backoff with jitter, but never shorter than the server's Retry-After."""
    delay = _backoff(state)
    exc = state.outcome.exception() if state.outcome else None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        delay = max(delay, min(retry_after, MAX_RETRY_AFTER_S))
    return delay


def _retry_after(response: requests.Response) -> float | None:
    """Read a Retry-After header given in seconds; dates or junk are ignored."""
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class FetchStats:
    """Responses downloaded, failed attempts that were retried, and answers from the cache."""

    downloads: int = 0
    retries: int = 0
    cache_hits: int = 0


class JolpicaClient:
    """Cache-aware client for Jolpica's Ergast-compatible endpoints.

    Use it as a context manager (``with JolpicaClient(cache) as client:``) so the network
    session is closed afterwards. Every argument after ``cache`` has a sensible default from
    ``config``; tests pass fakes for ``session``, ``limiter``, ``sleep`` and ``now``.
    """

    def __init__(
        self,
        cache: JsonCache,
        *,
        base_url: str | None = None,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
        timeout: tuple[float, float] | None = None,
        max_attempts: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
        refresh: bool = False,
    ) -> None:
        self.cache = cache
        self.base_url = (base_url or config.JOLPICA_BASE_URL).rstrip("/") + "/"
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self.limiter = limiter or RateLimiter(
            config.JOLPICA_MIN_INTERVAL_S,
            config.JOLPICA_MAX_CALLS_PER_HOUR,
            3600.0,
            max_interval=config.JOLPICA_MAX_INTERVAL_S,
            sleep=sleep,
        )
        self.timeout = timeout or config.JOLPICA_TIMEOUT_S
        self.refresh = refresh
        self.stats = FetchStats()
        self._now = now or (lambda: datetime.now(UTC))
        self._retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            wait=_wait_seconds,
            stop=stop_after_attempt(max_attempts or config.JOLPICA_MAX_ATTEMPTS),
            sleep=sleep,
            before_sleep=self._before_retry,
            reraise=True,
        )

    def __enter__(self) -> Self:
        """Return the client itself for use in a ``with`` block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the network session when the ``with`` block ends."""
        self.close()

    def close(self) -> None:
        """Close the underlying network session."""
        self.session.close()

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        fresh_since: datetime | None = None,
    ) -> dict[str, Any]:
        """Return one response, from the cache when possible.

        A cached copy is used when it was downloaded at or after ``fresh_since``; ``None``
        accepts any copy. Responses with zero rows are never cached, because "nothing yet"
        changes as soon as Jolpica publishes the session.
        """
        params = dict(params or {})
        if not self.refresh:
            entry = self.cache.load(path, params)
            if entry is not None and (fresh_since is None or entry.fetched_at >= fresh_since):
                self.stats.cache_hits += 1
                logger.debug("Cache hit: %s %s", path, params)
                return entry.payload
        try:
            url, payload = self._retrying(self._request, path, params)
        except requests.RequestException as exc:
            raise JolpicaError(f"Could not reach Jolpica for {path}: {exc}") from exc
        self.stats.downloads += 1
        self.limiter.record_success()
        if int(payload["MRData"].get("total", 0)) > 0:
            self.cache.save(path, params, url=url, payload=payload, fetched_at=self._now())
        return payload

    def get_races(
        self,
        path: str,
        *,
        results_key: str | None = None,
        fresh_since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch every page of an endpoint and return its races, merged across pages."""
        pages = []
        offset = 0
        while True:
            params = {"limit": config.JOLPICA_PAGE_LIMIT, "offset": offset}
            payload = self.get_json(path, params, fresh_since=fresh_since)
            pages.append(payload)
            info = payload["MRData"]
            total, limit = int(info["total"]), int(info["limit"])
            if limit < 1:
                raise JolpicaError(f"Jolpica returned a page limit of {limit} for {path}")
            offset += limit
            if offset >= total:
                return merge_pages(pages, results_key)

    def _request(self, path: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Send one GET request (after the rate limiter allows it); return the URL and JSON."""
        self.limiter.acquire()
        logger.debug("GET %s %s", path, params)
        response = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
        if response.status_code != 200:
            raise JolpicaHTTPError(response.status_code, response.url, _retry_after(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise JolpicaError(f"Jolpica returned invalid JSON for {response.url}") from exc
        if not isinstance(payload, dict) or "MRData" not in payload:
            raise JolpicaError(f"Unexpected response shape from {response.url}")
        return response.url, payload

    def _before_retry(self, state: RetryCallState) -> None:
        """Count and log a failed attempt; after an HTTP 429, also slow the rate limiter down."""
        self.stats.retries += 1
        exc = state.outcome.exception() if state.outcome else None
        wait = state.next_action.sleep if state.next_action else 0.0
        if isinstance(exc, JolpicaHTTPError) and exc.status == 429:
            self.limiter.slow_down()
            logger.info(
                "Jolpica asked us to slow down: now %.1f s between requests (retrying in %.1f s)",
                self.limiter.min_interval,
                wait,
            )
        else:
            logger.warning("Jolpica request failed (%s); retrying in %.1f s", exc, wait)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def merge_pages(
    pages: Iterable[dict[str, Any]], results_key: str | None = None
) -> list[dict[str, Any]]:
    """Combine the races from several pages into one list.

    Jolpica pages count result rows, not races, so one race can be split across two pages
    (for example 12 of its rows at the end of page 1 and 10 at the start of page 2). The two
    halves are joined back together under ``results_key``.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for payload in pages:
        for race in payload["MRData"]["RaceTable"].get("Races", []):
            key = (race["season"], race["round"])
            if key not in merged:
                merged[key] = dict(race)
                if results_key:
                    merged[key][results_key] = list(race.get(results_key, []))
            elif results_key:
                merged[key][results_key].extend(race.get(results_key, []))
    return list(merged.values())


# ---------------------------------------------------------------------------
# Season downloads
# ---------------------------------------------------------------------------


class RaceSource(Protocol):
    """Anything that can return merged races for an endpoint (the client, or a test fake)."""

    def get_races(
        self,
        path: str,
        *,
        results_key: str | None = None,
        fresh_since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return the races for ``path``, merged across pages."""
        ...


@dataclass
class SeasonData:
    """Raw Jolpica races for one season, one list per endpoint."""

    season: int
    schedule: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    qualifying: list[dict[str, Any]] = field(default_factory=list)
    sprint: list[dict[str, Any]] = field(default_factory=list)


def download_season(source: RaceSource, season: int, *, now: datetime) -> SeasonData:
    """Download a season's schedule, results, qualifying and sprint results.

    A finished season uses season-wide paginated requests, cached for good. The season in
    progress is fetched round by round, only for sessions that have already started, with
    the 72-hour freshness rule so late penalties are picked up.
    """
    if season < now.year:
        return _download_finished_season(source, season)
    return _download_current_season(source, season, now)


def fresh_since_for(session_start: datetime, now: datetime) -> datetime:
    """Earliest download time at which a cached copy of a session's results can be trusted.

    From 72 hours after the session, a copy downloaded after that point is final. Before
    then, results can still change, so any copy more than an hour old is downloaded again.
    """
    settled_at = session_start + timedelta(hours=config.RESULTS_SETTLE_HOURS)
    if now >= settled_at:
        return settled_at
    return now - timedelta(hours=config.RECENT_REFRESH_HOURS)


def _download_finished_season(source: RaceSource, season: int) -> SeasonData:
    """Season-wide requests for a season that is over; every page is cached permanently."""
    data = SeasonData(season, schedule=source.get_races(f"{season}.json"))
    for endpoint, results_key in RESULTS_KEYS.items():
        if endpoint == "sprint" and season < config.FIRST_SPRINT_SEASON:
            continue
        races = source.get_races(f"{season}/{endpoint}.json", results_key=results_key)
        setattr(data, endpoint, races)
    return data


def _download_current_season(source: RaceSource, season: int, now: datetime) -> SeasonData:
    """Round-by-round requests for the season in progress, skipping sessions not yet run."""
    schedule_since = now - timedelta(hours=config.SCHEDULE_REFRESH_HOURS)
    data = SeasonData(
        season, schedule=source.get_races(f"{season}.json", fresh_since=schedule_since)
    )
    for race in data.schedule:
        starts = session_starts(race)
        for endpoint, results_key in RESULTS_KEYS.items():
            start = starts[SESSION_OF_ENDPOINT[endpoint]]
            if start is None or start > now:
                continue
            races = source.get_races(
                f"{season}/{race['round']}/{endpoint}.json",
                results_key=results_key,
                fresh_since=fresh_since_for(start, now),
            )
            if not races:
                logger.info("%s round %s %s: not published yet", season, race["round"], endpoint)
            getattr(data, endpoint).extend(races)
    return data
