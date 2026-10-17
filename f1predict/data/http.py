"""Shared HTTP machinery for the API clients: disk cache, rate limiting, retries, timeouts.

Every request passes through three layers:

1. The disk cache. If a fresh enough copy is on disk, nothing is sent at all.
2. The rate limiter. Requests are spaced out, and the spacing widens automatically
   whenever the server answers HTTP 429 "too many requests".
3. Retries. Timeouts, dropped connections, HTTP 429 and HTTP 5xx are retried with
   exponential backoff; any other error fails straight away.

``JolpicaClient`` and ``OpenF1Client`` build on ``CachedJsonClient`` and only add what is
specific to their API: the response format, pagination, and what counts as "empty".
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

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

logger = logging.getLogger(__name__)

USER_AGENT = f"f1predict/{__version__} (+https://github.com/shrishrao/f1-race-predictor)"
MAX_RETRY_AFTER_S = 300.0
MIN_BACKOFF_INTERVAL_S = 0.5

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ApiError(RuntimeError):
    """An API could not be reached or returned something unusable."""


class ApiHTTPError(ApiError):
    """An API answered with an HTTP error status."""

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
    quarter, back toward the starting value. Real API limits can be stricter than the docs
    say and can change, so the limiter finds a safe speed by itself.

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

    def slow_down(self) -> None:
        """Double the gap between calls (at least 0.5 s, at most ``max_interval``)."""
        doubled = max(self.min_interval * 2, MIN_BACKOFF_INTERVAL_S)
        self.min_interval = min(doubled, self.max_interval)
        self._successes = 0

    def record_success(self) -> None:
        """Count a successful call; after enough in a row, shrink the gap by a quarter."""
        self._successes += 1
        if self._successes >= self.recover_after and self.min_interval > self.base_interval:
            self.min_interval = max(self.base_interval, self.min_interval * 0.75)
            self._successes = 0

    def _forget_up_to(self, cutoff: float) -> None:
        """Drop remembered calls made at or before ``cutoff``; they no longer count."""
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

_backoff = wait_exponential_jitter(initial=1, max=60)


def is_retryable(exc: BaseException) -> bool:
    """Retry timeouts, dropped connections, HTTP 429 (slow down) and HTTP 5xx (server trouble)."""
    if isinstance(exc, requests.Timeout | requests.ConnectionError):
        return True
    return isinstance(exc, ApiHTTPError) and (exc.status == 429 or exc.status >= 500)


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
# Freshness
# ---------------------------------------------------------------------------


def fresh_since_for(session_time: datetime, now: datetime) -> datetime:
    """Earliest download time at which a cached copy of a session's data can be trusted.

    From 72 hours after the session, a copy downloaded after that point is final. Before
    then, data can still change (penalties, late corrections), so any copy more than an
    hour old is downloaded again.
    """
    settled_at = session_time + timedelta(hours=config.RESULTS_SETTLE_HOURS)
    if now >= settled_at:
        return settled_at
    return now - timedelta(hours=config.RECENT_REFRESH_HOURS)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class FetchStats:
    """Responses downloaded, failed attempts that were retried, and answers from the cache."""

    downloads: int = 0
    retries: int = 0
    cache_hits: int = 0


class CachedJsonClient:
    """Base client: cache-aware GET requests with rate limiting, retries and timeouts.

    Subclasses set ``api_name`` and their error classes, and override ``is_empty``,
    ``check_payload`` or ``parse_response`` for their API. Use it as a context manager so
    the network session is closed afterwards.
    """

    api_name: ClassVar[str] = "API"
    error_class: ClassVar[type[ApiError]] = ApiError
    http_error_class: ClassVar[type[ApiHTTPError]] = ApiHTTPError

    def __init__(
        self,
        cache: JsonCache,
        *,
        base_url: str,
        limiter: RateLimiter,
        timeout: tuple[float, float],
        max_attempts: int,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
        refresh: bool = False,
    ) -> None:
        self.cache = cache
        self.base_url = base_url.rstrip("/") + "/"
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self.limiter = limiter
        self.timeout = timeout
        self.refresh = refresh
        self.stats = FetchStats()
        self._now = now or (lambda: datetime.now(UTC))
        self._retrying = Retrying(
            retry=retry_if_exception(is_retryable),
            wait=_wait_seconds,
            stop=stop_after_attempt(max_attempts),
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
        cache_empty: bool = False,
    ) -> Any:
        """Return one response, from the cache when possible.

        A cached copy is used when it was downloaded at or after ``fresh_since``; ``None``
        accepts any copy. Empty responses are only cached when ``cache_empty`` is set,
        because "nothing yet" usually changes once the data is published.
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
            raise self.error_class(f"Could not reach {self.api_name} for {path}: {exc}") from exc
        self.stats.downloads += 1
        self.limiter.record_success()
        if cache_empty or not self.is_empty(payload):
            self.cache.save(path, params, url=url, payload=payload, fetched_at=self._now())
        return payload

    def is_empty(self, payload: Any) -> bool:
        """Whether a response holds no data (subclasses know their API's format)."""
        return not payload

    def check_payload(self, payload: Any, url: str) -> None:
        """Raise if a decoded response doesn't have the expected shape (no check by default)."""

    def parse_response(self, response: requests.Response) -> Any:
        """Turn an HTTP response into decoded JSON, raising for error statuses and bad JSON."""
        if response.status_code != 200:
            raise self.http_error_class(response.status_code, response.url, _retry_after(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise self.error_class(
                f"{self.api_name} returned invalid JSON for {response.url}"
            ) from exc
        self.check_payload(payload, response.url)
        return payload

    def _request(self, path: str, params: dict[str, Any]) -> tuple[str, Any]:
        """Send one GET request (after the rate limiter allows it); return the URL and JSON."""
        self.limiter.acquire()
        logger.debug("GET %s%s %s", self.base_url, path, params)
        response = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
        return response.url, self.parse_response(response)

    def _before_retry(self, state: RetryCallState) -> None:
        """Count and log a failed attempt; after an HTTP 429, also slow the rate limiter down."""
        self.stats.retries += 1
        exc = state.outcome.exception() if state.outcome else None
        wait = state.next_action.sleep if state.next_action else 0.0
        if isinstance(exc, ApiHTTPError) and exc.status == 429:
            self.limiter.slow_down()
            logger.info(
                "%s asked us to slow down: now %.1f s between requests (retrying in %.1f s)",
                self.api_name,
                self.limiter.min_interval,
                wait,
            )
        else:
            logger.warning("%s request failed (%s); retrying in %.1f s", self.api_name, exc, wait)
