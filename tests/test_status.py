"""Tests for the finished / DNF / DNS / DSQ status classifier, across both Jolpica wordings."""

from __future__ import annotations

import pytest

from f1predict.data.status import StatusClass, classify_status


@pytest.mark.parametrize("status", ["Finished", "Lapped", "+1 Lap", "+2 Laps", "+12 Laps"])
def test_cars_that_took_the_flag_are_finishers(status: str) -> None:
    """Both wordings for a finished or lapped car count as finished."""
    assert classify_status(status) is StatusClass.FINISHED


def test_lapped_cars_from_2023_are_not_dnfs() -> None:
    """Regression guard: "Lapped" (2023 on) is a finisher, not a DNF (the brief's first rule)."""
    assert classify_status("Lapped") is StatusClass.FINISHED


@pytest.mark.parametrize(
    "status",
    ["Retired", "Engine", "Collision", "Collision damage", "Accident", "Undertray", "Wheel"],
)
def test_retirements_are_dnfs(status: str) -> None:
    """A generic "Retired" (2023 on) and a named cause (up to 2022) are both DNFs."""
    assert classify_status(status) is StatusClass.DNF


def test_unknown_status_counts_as_dnf() -> None:
    """Anything not recognized as finished, DNS or DSQ is treated as a retirement."""
    assert classify_status("Not classified") is StatusClass.DNF


@pytest.mark.parametrize("status", ["Did not start", "Withdrew"])
def test_non_starters_are_dns(status: str) -> None:
    """Drivers who never started are DNS, so they can be left out of training targets."""
    assert classify_status(status) is StatusClass.DNS


@pytest.mark.parametrize("status", ["Disqualified", "Excluded"])
def test_disqualifications_are_dsq(status: str) -> None:
    """Disqualified drivers are DSQ, kept apart from both finishers and DNFs."""
    assert classify_status(status) is StatusClass.DSQ


def test_surrounding_whitespace_is_ignored() -> None:
    """Stray spaces around a status do not change its class."""
    assert classify_status("  Finished ") is StatusClass.FINISHED
