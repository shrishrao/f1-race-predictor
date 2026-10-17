"""Tests for the shared HTTP machinery: the adaptive rate limiter and freshness rules.

Time is faked with ``FakeClock``, so a test of "wait 70 seconds" runs instantly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from f1predict.data.http import RateLimiter, fresh_since_for


class FakeClock:
    """A clock that only moves when the code under test sleeps (or the test moves it)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        """Return the current fake time in seconds."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record the sleep and jump the clock forward instead of waiting."""
        self.sleeps.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------------------
# Fixed spacing and the sliding window
# ---------------------------------------------------------------------------


def test_back_to_back_calls_are_spaced_out() -> None:
    """Three instant calls with a 0.3 s minimum gap wait 0.3 s twice."""
    clock = FakeClock()
    limiter = RateLimiter(0.3, 100, 3600.0, clock=clock.time, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
    assert clock.sleeps == pytest.approx([0.3, 0.3])


def test_no_wait_when_calls_are_already_far_apart() -> None:
    """Calls that are naturally a second apart never wait."""
    clock = FakeClock()
    limiter = RateLimiter(0.3, 100, 3600.0, clock=clock.time, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
        clock.now += 1.0
    assert clock.sleeps == []


def test_full_window_waits_until_the_oldest_call_expires() -> None:
    """With 3 calls allowed per 100 s, the 4th and 5th calls wait for slots to free up."""
    clock = FakeClock()
    limiter = RateLimiter(0.0, 3, 100.0, clock=clock.time, sleep=clock.sleep)
    for moment in (0.0, 10.0, 20.0):
        clock.now = moment
        limiter.acquire()
    clock.now = 30.0
    limiter.acquire()
    assert clock.sleeps == [70.0]
    limiter.acquire()
    assert clock.sleeps == [70.0, 10.0]


def test_limiter_needs_a_positive_budget() -> None:
    """A budget of zero calls is a configuration mistake."""
    with pytest.raises(ValueError):
        RateLimiter(0.3, 0, 3600.0)


# ---------------------------------------------------------------------------
# Adapting to HTTP 429
# ---------------------------------------------------------------------------


def test_slow_down_doubles_the_gap_up_to_a_cap() -> None:
    """Each "too many requests" doubles the gap: 0.5 -> 1 -> 2 -> 4 -> 8, then stays at 8."""
    limiter = RateLimiter(0.5, 100, 3600.0, max_interval=8.0)
    gaps = []
    for _ in range(5):
        limiter.slow_down()
        gaps.append(limiter.min_interval)
    assert gaps == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_gap_recovers_after_a_run_of_successes_but_not_below_the_start() -> None:
    """Every 20 successes in a row shrink the gap by a quarter, never below the starting gap."""
    limiter = RateLimiter(0.5, 100, 3600.0, recover_after=20)
    limiter.slow_down()
    limiter.slow_down()
    for _ in range(19):
        limiter.record_success()
    assert limiter.min_interval == 2.0
    limiter.record_success()
    assert limiter.min_interval == 1.5
    for _ in range(200):
        limiter.record_success()
    assert limiter.min_interval == 0.5


def test_a_zero_gap_limiter_still_backs_off() -> None:
    """Even with no starting gap, a slow-down waits at least half a second."""
    limiter = RateLimiter(0.0, 100, 3600.0)
    limiter.slow_down()
    assert limiter.min_interval == 0.5


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def test_fresh_since_for_switches_at_72_hours() -> None:
    """Before 72 h, copies over an hour old are stale; after, only copies made post-72 h count."""
    start = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
    before = start + timedelta(hours=71)
    after = start + timedelta(hours=80)
    assert fresh_since_for(start, before) == before - timedelta(hours=1)
    assert fresh_since_for(start, after) == start + timedelta(hours=72)
