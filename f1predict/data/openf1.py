"""OpenF1 API client and practice-session downloads.

OpenF1 (https://openf1.org) keeps its own copy of F1's timing data and serves it without a
login from 2023 on (data licensed CC BY-NC-SA 4.0, non-commercial). Each practice session
needs four requests: the driver list, the laps, the stints (tyre sets) and the race-control
messages. OpenF1 sessions are matched to Jolpica rounds by session number and start time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
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

logger = logging.getLogger(__name__)

PRACTICE_SESSIONS = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3"}
SESSION_ENDPOINTS = ("drivers", "laps", "stints", "race_control")
MATCH_TOLERANCE = timedelta(hours=36)
AGREEMENT_TOLERANCE = timedelta(hours=3)
NO_RESULTS = "No results found."

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OpenF1Error(ApiError):
    """The OpenF1 API could not be reached or returned something unusable."""


class OpenF1HTTPError(OpenF1Error, ApiHTTPError):
    """OpenF1 answered with an HTTP error status."""


class PracticeAlignmentError(ValueError):
    """An OpenF1 practice session could not be matched to exactly one Jolpica round."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenF1Client(CachedJsonClient):
    """Cache-aware client for OpenF1's REST endpoints (``sessions``, ``laps``, ...).

    Every argument after ``cache`` has a default from ``config``; tests pass fakes for
    ``session``, ``limiter``, ``sleep`` and ``now``.
    """

    api_name = "OpenF1"
    error_class = OpenF1Error
    http_error_class = OpenF1HTTPError

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
            base_url=base_url or config.OPENF1_BASE_URL,
            limiter=limiter
            or RateLimiter(
                config.OPENF1_MIN_INTERVAL_S,
                config.OPENF1_MAX_CALLS_PER_MINUTE,
                60.0,
                max_interval=config.OPENF1_MAX_INTERVAL_S,
                sleep=sleep,
            ),
            timeout=timeout or config.OPENF1_TIMEOUT_S,
            max_attempts=max_attempts or config.OPENF1_MAX_ATTEMPTS,
            session=session,
            sleep=sleep,
            now=now,
            refresh=refresh,
        )

    def parse_response(self, response: requests.Response) -> Any:
        """Like the base version, but OpenF1's 404 "No results found." means an empty list."""
        if response.status_code == 404 and _says_no_results(response):
            return []
        return super().parse_response(response)

    def check_payload(self, payload: Any, url: str) -> None:
        """Every OpenF1 response is a JSON list of records."""
        if not isinstance(payload, list):
            raise OpenF1Error(f"Unexpected response shape from {url}")


def _says_no_results(response: requests.Response) -> bool:
    """Whether a response body is OpenF1's ``{"detail": "No results found."}``."""
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("detail") == NO_RESULTS


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PracticeSession:
    """One OpenF1 practice session, matched to its Jolpica round."""

    season: int
    round: int
    session: str
    session_key: int
    start: datetime
    end: datetime


@dataclass
class PracticeData:
    """The raw OpenF1 records for one practice session."""

    session: PracticeSession
    drivers: list[dict[str, Any]]
    laps: list[dict[str, Any]]
    stints: list[dict[str, Any]]
    race_control: list[dict[str, Any]]


def match_practice_sessions(
    sessions: list[dict[str, Any]], schedule: pd.DataFrame, season: int
) -> list[PracticeSession]:
    """Pair each OpenF1 practice session with its Jolpica round, by session number and start.

    Pre-season testing ("Day 1", ...) and cancelled sessions are skipped. Every other
    session must line up with exactly one round's FP1/FP2/FP3 start within 36 hours, and no
    round's session may be claimed twice; anything else raises ``PracticeAlignmentError``,
    because a silent mismatch would attach practice pace to the wrong race. The window is
    that wide because the sources sometimes disagree by a whole day: Jolpica lists the 2023
    Las Vegas sessions one day early (local date with a UTC time). Gaps over three hours are
    logged as warnings.
    """
    rounds = schedule[schedule["season"] == season]
    matched = []
    for record in sessions:
        name = record.get("session_name")
        if name not in PRACTICE_SESSIONS or record.get("is_cancelled"):
            continue
        label = PRACTICE_SESSIONS[name]
        start = pd.Timestamp(record["date_start"]).tz_convert("UTC")
        gap = (rounds[f"{label.lower()}_start"] - start).abs()
        within = gap.le(MATCH_TOLERANCE).fillna(False)
        hits = rounds.loc[within, "round"]
        if len(hits) != 1:
            raise PracticeAlignmentError(
                f"{season} {name} starting {start:%Y-%m-%d %H:%M} UTC "
                f"matches {len(hits)} Jolpica rounds (expected exactly 1)"
            )
        if gap[within].iloc[0] > AGREEMENT_TOLERANCE:
            logger.warning(
                "%s round %s %s: OpenF1 starts it at %s UTC but Jolpica %.1f hours away; "
                "matched anyway",
                season,
                int(hits.iloc[0]),
                label,
                f"{start:%Y-%m-%d %H:%M}",
                gap[within].iloc[0].total_seconds() / 3600,
            )
        matched.append(
            PracticeSession(
                season=season,
                round=int(hits.iloc[0]),
                session=label,
                session_key=int(record["session_key"]),
                start=start.to_pydatetime(),
                end=pd.Timestamp(record["date_end"]).tz_convert("UTC").to_pydatetime(),
            )
        )
    claimed = [(m.round, m.session) for m in matched]
    repeated = sorted({key for key in claimed if claimed.count(key) > 1})
    if repeated:
        raise PracticeAlignmentError(f"{season}: several OpenF1 sessions match {repeated}")
    return sorted(matched, key=lambda m: (m.round, m.session))


def download_practice_season(
    client: CachedJsonClient, season: int, schedule: pd.DataFrame, *, now: datetime
) -> list[PracticeData]:
    """Download every finished practice session of a season, four requests per session.

    Like Jolpica results, a session's data only counts as final once it was downloaded 72
    hours after the session; from then on even an empty answer is cached, so sessions that
    genuinely have no data are not asked for again on every run.
    """
    since = None if season < now.year else now - timedelta(hours=config.SCHEDULE_REFRESH_HOURS)
    params = {"year": season, "session_type": "Practice"}
    sessions = client.get_json("sessions", params, fresh_since=since)
    downloaded = []
    for session in match_practice_sessions(sessions, schedule, season):
        if session.end > now:
            continue
        settled = now >= session.end + timedelta(hours=config.RESULTS_SETTLE_HOURS)
        fresh = fresh_since_for(session.end, now)
        before = client.stats.downloads
        records = {
            endpoint: client.get_json(
                endpoint,
                {"session_key": session.session_key},
                fresh_since=fresh,
                cache_empty=settled,
            )
            for endpoint in SESSION_ENDPOINTS
        }
        if client.stats.downloads > before:
            logger.info(
                "%s round %s %s: %s laps downloaded",
                season,
                session.round,
                session.session,
                len(records["laps"]),
            )
        downloaded.append(PracticeData(session=session, **records))
    return downloaded
