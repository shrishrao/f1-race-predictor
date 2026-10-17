"""The pipeline behind ``python -m f1predict fetch``: download, tidy, check, save.

Two parts, run in this order: Jolpica (schedules, results, qualifying, sprints) and then
OpenF1 practice laps, which need Jolpica's schedule and entry lists to line up with.
Every response is cached as soon as it arrives, so a run that stops halfway (network drop,
Ctrl+C) picks up where it left off when started again.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from f1predict import config
from f1predict.data.cache import JsonCache
from f1predict.data.jolpica import JolpicaClient, download_season
from f1predict.data.mapping import round_entries
from f1predict.data.openf1 import OpenF1Client, download_practice_season
from f1predict.data.practice import build_practice_season
from f1predict.data.tables import build_tables, load_table, save_season_tables, validate_season

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jolpica seasons
# ---------------------------------------------------------------------------


@dataclass
class SeasonSummary:
    """What one season's fetch produced, for the summary table and the log."""

    season: int
    races: int
    results: int
    qualifying: int
    sprint: int
    downloads: int
    retries: int
    cache_hits: int
    warnings: list[str] = field(default_factory=list)


def fetch_jolpica(
    first: int,
    last: int,
    *,
    refresh: bool = False,
    now: datetime | None = None,
    client: JolpicaClient | None = None,
) -> list[SeasonSummary]:
    """Download seasons ``first``..``last`` from Jolpica and save tidy tables for each.

    Every response is cached as soon as it arrives, so if the run stops halfway (network
    drop, Ctrl+C), running it again picks up where it left off without re-downloading.
    """
    now = now or datetime.now(UTC)
    client = client or JolpicaClient(JsonCache(config.JOLPICA_CACHE_DIR), refresh=refresh)
    summaries = []
    with client:
        for season in range(first, last + 1):
            before = dataclasses.replace(client.stats)
            data = download_season(client, season, now=now)
            tables = build_tables(
                schedule=data.schedule,
                results=data.results,
                qualifying=data.qualifying,
                sprint=data.sprint,
            )
            warnings = validate_season(tables, season, now=now)
            counts = save_season_tables(tables, season, config.INTERIM_DIR)
            summary = SeasonSummary(
                season=season,
                races=counts["schedule"],
                results=counts["results"],
                qualifying=counts["qualifying"],
                sprint=counts["sprint"],
                downloads=client.stats.downloads - before.downloads,
                retries=client.stats.retries - before.retries,
                cache_hits=client.stats.cache_hits - before.cache_hits,
                warnings=warnings,
            )
            logger.info(
                "%s: %s races, %s results, %s qualifying, %s sprint rows "
                "(%s downloaded, %s retries, %s cached)",
                season,
                summary.races,
                summary.results,
                summary.qualifying,
                summary.sprint,
                summary.downloads,
                summary.retries,
                summary.cache_hits,
            )
            summaries.append(summary)
    return summaries


# ---------------------------------------------------------------------------
# OpenF1 practice laps
# ---------------------------------------------------------------------------


@dataclass
class PracticeSummary:
    """What one season's practice fetch produced, for the summary table and the log."""

    season: int
    sessions: int
    laps: int
    drivers: int
    practice_only: int
    downloads: int
    retries: int
    cache_hits: int
    warnings: list[str] = field(default_factory=list)


def practice_seasons(first: int, last: int) -> range:
    """The seasons in ``first``..``last`` that have OpenF1 practice data (2023 on)."""
    return range(max(first, config.FIRST_PRACTICE_SEASON), last + 1)


def fetch_practice(
    first: int,
    last: int,
    *,
    refresh: bool = False,
    now: datetime | None = None,
    client: OpenF1Client | None = None,
) -> list[PracticeSummary]:
    """Download practice laps for seasons ``first``..``last`` (2023 on) and save tidy tables.

    Needs the Jolpica tables for those seasons, so run it after ``fetch_jolpica``. Writes
    ``practice/<season>.parquet`` (one row per lap) and ``practice_only/<season>.parquet``
    (FP1 stand-ins who never raced that weekend).
    """
    seasons = practice_seasons(first, last)
    if not seasons:
        return []
    now = now or datetime.now(UTC)
    schedule = load_table("schedule")
    entries = round_entries(*(_load_or_empty(name) for name in ("results", "qualifying", "sprint")))
    client = client or OpenF1Client(JsonCache(config.OPENF1_CACHE_DIR), refresh=refresh)
    summaries = []
    with client:
        for season in seasons:
            before = dataclasses.replace(client.stats)
            sessions = download_practice_season(client, season, schedule, now=now)
            laps, stand_ins, warnings = build_practice_season(
                sessions, entries[entries["season"] == season]
            )
            save_season_tables({"practice": laps, "practice_only": stand_ins}, season)
            summary = PracticeSummary(
                season=season,
                sessions=len(sessions),
                laps=len(laps),
                drivers=int(laps["driver_id"].nunique()),
                practice_only=len(stand_ins),
                downloads=client.stats.downloads - before.downloads,
                retries=client.stats.retries - before.retries,
                cache_hits=client.stats.cache_hits - before.cache_hits,
                warnings=warnings,
            )
            logger.info(
                "%s practice: %s sessions, %s laps, %s drivers, %s stand-in entries "
                "(%s downloaded, %s retries, %s cached)",
                season,
                summary.sessions,
                summary.laps,
                summary.drivers,
                summary.practice_only,
                summary.downloads,
                summary.retries,
                summary.cache_hits,
            )
            summaries.append(summary)
    return summaries


def _load_or_empty(name: str) -> pd.DataFrame:
    """Load a saved table, or an empty one if it was never fetched (e.g. no sprints yet)."""
    try:
        return load_table(name)
    except FileNotFoundError:
        return pd.DataFrame()
