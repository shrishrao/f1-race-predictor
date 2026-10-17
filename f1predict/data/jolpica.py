"""Jolpica F1 API client and season downloads.

The shared machinery (disk cache, rate limiter, retries, timeouts) lives in
``f1predict.data.http``. This module adds what is specific to Jolpica: its response format
(everything is wrapped in ``MRData``), its pagination, and which requests a season needs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

import requests

from f1predict import config
from f1predict.data.cache import JsonCache
from f1predict.data.http import (
    ApiError,
    ApiHTTPError,
    CachedJsonClient,
    RateLimiter,
    fresh_since_for,
)
from f1predict.data.tables import session_starts

logger = logging.getLogger(__name__)

RESULTS_KEYS = {
    "results": "Results",
    "qualifying": "QualifyingResults",
    "sprint": "SprintResults",
}
SESSION_OF_ENDPOINT = {"results": "race", "qualifying": "quali", "sprint": "sprint"}

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class JolpicaError(ApiError):
    """The Jolpica API could not be reached or returned something unusable."""


class JolpicaHTTPError(JolpicaError, ApiHTTPError):
    """Jolpica answered with an HTTP error status."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class JolpicaClient(CachedJsonClient):
    """Cache-aware client for Jolpica's Ergast-compatible endpoints.

    Every argument after ``cache`` has a default from ``config``; tests pass fakes for
    ``session``, ``limiter``, ``sleep`` and ``now``.
    """

    api_name = "Jolpica"
    error_class = JolpicaError
    http_error_class = JolpicaHTTPError

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
        super().__init__(
            cache,
            base_url=base_url or config.JOLPICA_BASE_URL,
            limiter=limiter
            or RateLimiter(
                config.JOLPICA_MIN_INTERVAL_S,
                config.JOLPICA_MAX_CALLS_PER_HOUR,
                3600.0,
                max_interval=config.JOLPICA_MAX_INTERVAL_S,
                sleep=sleep,
            ),
            timeout=timeout or config.JOLPICA_TIMEOUT_S,
            max_attempts=max_attempts or config.JOLPICA_MAX_ATTEMPTS,
            session=session,
            sleep=sleep,
            now=now,
            refresh=refresh,
        )

    def is_empty(self, payload: Any) -> bool:
        """A Jolpica response is empty when its row count (``MRData.total``) is zero."""
        return int(payload["MRData"].get("total", 0)) == 0

    def check_payload(self, payload: Any, url: str) -> None:
        """Every Jolpica response is a JSON object with an ``MRData`` key."""
        if not isinstance(payload, dict) or "MRData" not in payload:
            raise JolpicaError(f"Unexpected response shape from {url}")

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
