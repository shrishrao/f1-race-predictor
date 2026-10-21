"""Project-wide settings: file locations, seasons, and default simulation settings.

Two environment variables can move the big folders, e.g. to another drive:

- ``F1PREDICT_DATA_DIR``: raw caches, intermediate tables, features, published artifacts.
- ``F1PREDICT_MODELS_DIR``: trained model files.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dir_from_env(var: str, default: Path) -> Path:
    """Return the folder named by environment variable ``var``, or ``default`` if unset/blank."""
    value = os.environ.get(var, "").strip()
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _dir_from_env("F1PREDICT_DATA_DIR", REPO_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
JOLPICA_CACHE_DIR = RAW_DIR / "jolpica"
OPENF1_CACHE_DIR = RAW_DIR / "openf1"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PUBLISHED_DIR = DATA_DIR / "published"
LOG_DIR = DATA_DIR / "logs"
MODELS_DIR = _dir_from_env("F1PREDICT_MODELS_DIR", REPO_ROOT / "models")
DOCS_DIR = REPO_ROOT / "docs"
README_PATH = REPO_ROOT / "README.md"

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

FIRST_SEASON = 2018
LAST_SEASON = 2026
VALIDATION_SEASON = 2024
TEST_SEASONS = (2025, 2026)

# Races left out of training, evaluation and rolling form, with the reason.
EXCLUDED_RACES = {
    (2021, 12): "Belgian GP: one lap behind the safety car, so the result is the grid",
}

# ---------------------------------------------------------------------------
# Outcomes
# Every model gives each driver a chance of finishing in the top 1, 3 and 10 places, read
# from the official classification (a disqualified car loses its place).
# ---------------------------------------------------------------------------

OUTCOMES = {"win": 1, "podium": 3, "points": 10}

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_SIMS = 100_000

# ---------------------------------------------------------------------------
# Jolpica API
# The docs allow bursts of 4 requests/second and 500 requests/hour, and `limit` is capped
# at 100 rows per request. In practice Jolpica throttles sooner (HTTP 429 after roughly
# 70 quick requests), so the client starts at one request per 0.5 s and doubles the gap
# after every 429, up to 8 s.
# ---------------------------------------------------------------------------

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1/"
JOLPICA_PAGE_LIMIT = 100
JOLPICA_MIN_INTERVAL_S = 0.5
JOLPICA_MAX_INTERVAL_S = 8.0
JOLPICA_MAX_CALLS_PER_HOUR = 450
JOLPICA_TIMEOUT_S = (5.0, 30.0)
JOLPICA_MAX_ATTEMPTS = 6
FIRST_SPRINT_SEASON = 2021

# ---------------------------------------------------------------------------
# OpenF1 API (practice laps)
# F1's own timing archive needs an F1 TV login since October 2026, so practice laps come
# from OpenF1, which serves them without one from 2023 on. Its free tier allows 3 requests
# per second and 30 per minute; the client stays at 28 per minute.
# ---------------------------------------------------------------------------

OPENF1_BASE_URL = "https://api.openf1.org/v1/"
OPENF1_MIN_INTERVAL_S = 0.4
OPENF1_MAX_INTERVAL_S = 8.0
OPENF1_MAX_CALLS_PER_MINUTE = 28
OPENF1_TIMEOUT_S = (5.0, 60.0)
OPENF1_MAX_ATTEMPTS = 6
FIRST_PRACTICE_SEASON = 2023

# ---------------------------------------------------------------------------
# Cache freshness
# Results can change for hours after a session (penalties, disqualifications), so a
# copy only counts as final once it was downloaded 72 hours after the session started.
# Until then it is re-checked at most once an hour. The current season's calendar is
# re-checked daily; finished seasons are never downloaded twice.
# ---------------------------------------------------------------------------

RESULTS_SETTLE_HOURS = 72
RECENT_REFRESH_HOURS = 1
SCHEDULE_REFRESH_HOURS = 24
