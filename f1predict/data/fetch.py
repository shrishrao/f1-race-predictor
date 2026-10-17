"""The pipeline behind ``python -m f1predict fetch``: download, tidy, check, save.

Phase 2 covers the Jolpica part; the FastF1 practice download joins in Phase 3.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from f1predict import config
from f1predict.data.cache import JsonCache
from f1predict.data.jolpica import JolpicaClient, download_season
from f1predict.data.tables import build_tables, save_season_tables, validate_season

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
