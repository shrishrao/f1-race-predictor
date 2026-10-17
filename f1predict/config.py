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
FASTF1_CACHE_DIR = RAW_DIR / "fastf1"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PUBLISHED_DIR = DATA_DIR / "published"
LOG_DIR = DATA_DIR / "logs"
MODELS_DIR = _dir_from_env("F1PREDICT_MODELS_DIR", REPO_ROOT / "models")
DOCS_DIR = REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

FIRST_SEASON = 2018
LAST_SEASON = 2026
VALIDATION_SEASON = 2024
TEST_SEASONS = (2025, 2026)

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_SIMS = 100_000

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1/"
