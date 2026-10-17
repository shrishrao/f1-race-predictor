"""Turn raw Jolpica races into tidy tables, check them, and save them as parquet files.

Tidy means one row per thing: one row per race weekend in ``schedule``, and one row per
driver per race in ``results``, ``qualifying`` and ``sprint``. Each season is saved as its
own file (for example ``data/interim/results/2023.parquet``), so downloading one season
again never touches the others.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from f1predict import config
from f1predict.data.status import classify_status

logger = logging.getLogger(__name__)

TABLE_NAMES = ("schedule", "results", "qualifying", "sprint")
SESSIONS = ("fp1", "fp2", "fp3", "sprint_quali", "sprint", "quali", "race")
SESSION_KEYS = {
    "FirstPractice": "fp1",
    "SecondPractice": "fp2",
    "ThirdPractice": "fp3",
    "SprintShootout": "sprint_quali",
    "SprintQualifying": "sprint_quali",
    "Sprint": "sprint",
    "Qualifying": "quali",
}
MIN_ENTRIES_PER_RACE = 18
PUBLISH_GRACE_HOURS = 24
_LAP_TIME = re.compile(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")

# ---------------------------------------------------------------------------
# Column types (used for every table, including empty ones)
# ---------------------------------------------------------------------------

SCHEDULE_DTYPES = {
    "season": "int64",
    "round": "int64",
    "race_name": "string",
    "circuit_id": "string",
    "circuit_name": "string",
    "locality": "string",
    "country": "string",
    **{f"{session}_start": "datetime64[ns, UTC]" for session in SESSIONS},
    "is_sprint_weekend": "bool",
}
RESULT_DTYPES = {
    "season": "int64",
    "round": "int64",
    "driver_id": "string",
    "driver_code": "string",
    "driver_name": "string",
    "car_number": "Int64",
    "constructor_id": "string",
    "constructor_name": "string",
    "grid": "Int64",
    "position": "Int64",
    "position_text": "string",
    "points": "float64",
    "laps": "Int64",
    "status": "string",
    "status_class": "string",
    "time_ms": "Int64",
}
QUALIFYING_DTYPES = {
    "season": "int64",
    "round": "int64",
    "driver_id": "string",
    "driver_code": "string",
    "car_number": "Int64",
    "constructor_id": "string",
    "position": "Int64",
    "q1_s": "float64",
    "q2_s": "float64",
    "q3_s": "float64",
}


class DataValidationError(ValueError):
    """Downloaded data failed a sanity check that real, complete data should always pass."""


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------


def parse_utc(date_text: str | None, time_text: str | None) -> datetime | None:
    """Combine Jolpica's ``date`` and ``time`` fields into a UTC datetime.

    Seasons before 2022 give only a date for practice sessions; those get 12:00 UTC as an
    approximate time. Returns None when there is no date at all.
    """
    if not date_text:
        return None
    stamp = datetime.fromisoformat(f"{date_text}T{time_text or '12:00:00Z'}")
    return stamp.astimezone(UTC) if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def session_starts(race: dict[str, Any]) -> dict[str, datetime | None]:
    """Return the UTC start of every session in a race weekend (None if the weekend lacks it).

    Session names are normalized across formats: the 2023 "SprintShootout" and the 2024+
    "SprintQualifying" both become ``sprint_quali``.
    """
    starts: dict[str, datetime | None] = dict.fromkeys(SESSIONS)
    for key, session in SESSION_KEYS.items():
        block = race.get(key)
        if isinstance(block, dict):
            starts[session] = parse_utc(block.get("date"), block.get("time"))
    starts["race"] = parse_utc(race.get("date"), race.get("time"))
    return starts


def parse_lap_time(text: str | None) -> float:
    """Convert a lap time like ``"1:23.456"`` to seconds; blanks and anything odd become NaN."""
    match = _LAP_TIME.match(text.strip()) if text else None
    if match is None:
        return math.nan
    return int(match.group(1) or 0) * 60 + float(match.group(2))


def _to_int(value: Any) -> int | None:
    """Parse an integer field that may be missing or blank."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _frame(rows: list[dict[str, Any]], dtypes: dict[str, str]) -> pd.DataFrame:
    """Build a DataFrame with exactly these columns and types, even when there are no rows."""
    frame = pd.DataFrame(rows, columns=list(dtypes))
    for column, dtype in dtypes.items():
        if dtype.startswith("datetime64"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        else:
            frame[column] = frame[column].astype(dtype)
    return frame


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def schedule_table(races: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """One row per race weekend: name, circuit, and the UTC start time of every session."""
    rows = []
    for race in races:
        circuit = race.get("Circuit", {})
        location = circuit.get("Location", {})
        starts = session_starts(race)
        rows.append(
            {
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race.get("raceName"),
                "circuit_id": circuit.get("circuitId"),
                "circuit_name": circuit.get("circuitName"),
                "locality": location.get("locality"),
                "country": location.get("country"),
                **{f"{session}_start": starts[session] for session in SESSIONS},
                "is_sprint_weekend": starts["sprint"] is not None,
            }
        )
    return _frame(rows, SCHEDULE_DTYPES)


def results_table(races: Iterable[dict[str, Any]], results_key: str = "Results") -> pd.DataFrame:
    """One row per driver per race (or sprint): grid, position, points and finishing status.

    ``car_number`` is the number the car actually raced with that weekend, which is what
    FastF1 uses. Jolpica's ``permanentNumber`` is the driver's current number and is wrong
    for past seasons (Verstappen raced #33 and #1, but it says 3), so it is not kept.
    """
    rows = []
    for race in races:
        for entry in race.get(results_key, []):
            driver = entry.get("Driver", {})
            constructor = entry.get("Constructor", {})
            status = entry.get("status", "")
            name = f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "driver_name": name,
                    "car_number": _to_int(entry.get("number")),
                    "constructor_id": constructor.get("constructorId"),
                    "constructor_name": constructor.get("name"),
                    "grid": _to_int(entry.get("grid")),
                    "position": _to_int(entry.get("position")),
                    "position_text": entry.get("positionText"),
                    "points": float(entry.get("points") or 0.0),
                    "laps": _to_int(entry.get("laps")),
                    "status": status,
                    "status_class": classify_status(status).value,
                    "time_ms": _to_int((entry.get("Time") or {}).get("millis")),
                }
            )
    return _frame(rows, RESULT_DTYPES)


def qualifying_table(races: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """One row per driver per qualifying session, with Q1/Q2/Q3 times in seconds (NaN if none)."""
    rows = []
    for race in races:
        for entry in race.get("QualifyingResults", []):
            driver = entry.get("Driver", {})
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "car_number": _to_int(entry.get("number")),
                    "constructor_id": entry.get("Constructor", {}).get("constructorId"),
                    "position": _to_int(entry.get("position")),
                    "q1_s": parse_lap_time(entry.get("Q1")),
                    "q2_s": parse_lap_time(entry.get("Q2")),
                    "q3_s": parse_lap_time(entry.get("Q3")),
                }
            )
    return _frame(rows, QUALIFYING_DTYPES)


def build_tables(
    *,
    schedule: Iterable[dict[str, Any]],
    results: Iterable[dict[str, Any]],
    qualifying: Iterable[dict[str, Any]],
    sprint: Iterable[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Turn one season of raw Jolpica races into the four tidy tables."""
    return {
        "schedule": schedule_table(schedule),
        "results": results_table(results, "Results"),
        "qualifying": qualifying_table(qualifying),
        "sprint": results_table(sprint, "SprintResults"),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_season(tables: dict[str, pd.DataFrame], season: int, *, now: datetime) -> list[str]:
    """Sanity-check one season and return warnings (or raise for problems that must not happen).

    Always an error: an empty schedule, or duplicate rows. For a finished season, these are
    errors too: a scheduled round without results or qualifying, a sprint weekend without
    sprint results, or a race with fewer than 18 entries. For the season in progress the
    same gaps are only warnings, for sessions that ended at least 24 hours ago, because
    Jolpica may simply not have published them yet.
    """
    schedule = tables["schedule"]
    if schedule.empty:
        raise DataValidationError(f"{season}: the schedule is empty")
    for name, keys in (
        ("schedule", ["round"]),
        ("results", ["round", "driver_id"]),
        ("qualifying", ["round", "driver_id"]),
        ("sprint", ["round", "driver_id"]),
    ):
        duplicates = int(tables[name].duplicated(keys).sum())
        if duplicates:
            raise DataValidationError(f"{season} {name}: {duplicates} duplicate rows")

    finished = season < now.year
    cutoff = pd.Timestamp(now - timedelta(hours=PUBLISH_GRACE_HOURS))
    expected = {
        "results": schedule["race_start"],
        "qualifying": schedule["quali_start"],
        "sprint": schedule["sprint_start"],
    }
    problems = []
    for name, starts in expected.items():
        due = starts.notna() if finished else starts.notna() & (starts <= cutoff)
        have = {int(r) for r in tables[name]["round"]}
        missing = sorted(int(r) for r in schedule.loc[due, "round"] if int(r) not in have)
        if missing:
            problems.append(f"{season} {name}: no data for rounds {missing}")
    counts = tables["results"].groupby("round").size()
    short = sorted(int(r) for r in counts[counts < MIN_ENTRIES_PER_RACE].index)
    if short:
        problems.append(
            f"{season} results: fewer than {MIN_ENTRIES_PER_RACE} entries in rounds {short}"
        )

    if problems and finished:
        raise DataValidationError("; ".join(problems))
    for problem in problems:
        logger.warning(problem)
    return problems


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def save_season_tables(
    tables: dict[str, pd.DataFrame], season: int, interim_dir: Path | None = None
) -> dict[str, int]:
    """Write each table to ``<interim>/<table>/<season>.parquet`` and return the row counts.

    Files are written to a temporary name and then renamed, so a crash never leaves a broken
    parquet file. An empty table removes any old file for that season.
    """
    root = interim_dir or config.INTERIM_DIR
    counts = {}
    for name, frame in tables.items():
        counts[name] = len(frame)
        target = root / name / f"{season}.parquet"
        if frame.empty:
            target.unlink(missing_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, target)
    return counts


def load_table(name: str, interim_dir: Path | None = None) -> pd.DataFrame:
    """Read every saved season of one table into a single DataFrame, sorted by season and round."""
    if name not in TABLE_NAMES:
        raise ValueError(f"Unknown table {name!r}; expected one of {TABLE_NAMES}")
    folder = (interim_dir or config.INTERIM_DIR) / name
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No {name} data in {folder}. Run `python -m f1predict fetch` first."
        )
    frame = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    return frame.sort_values(["season", "round"], kind="stable", ignore_index=True)
