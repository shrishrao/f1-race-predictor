"""Tidy practice laps from OpenF1: one row per timed lap, with the flags features need.

OpenF1 gives lap times, stints (tyre sets) and race-control messages separately. This
module joins them and works out, for every lap:

- the stint number, tyre compound and tyre age;
- whether it is an out-lap (``is_pit_out_lap``) or an in-lap (the last lap of a stint);
- whether race control deleted its time (track limits), matched by lap time;
- whether the track was green the whole lap (no yellow, red, VSC or safety car);
- whether it is "accurate": all three sector times present and adding up to the lap time.

Drivers are then mapped to Jolpica ids round by round (see ``mapping``). Stand-ins keep
their laps with an empty ``driver_id``, so they still count when finding a session's
fastest lap, and are listed in the ``practice_only`` audit table.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import pandas as pd

from f1predict.data.mapping import map_session_drivers
from f1predict.data.openf1 import PracticeData
from f1predict.data.tables import typed_frame

logger = logging.getLogger(__name__)

TIME_MATCH_TOLERANCE_S = 0.0015
SECTOR_SUM_TOLERANCE_S = 0.01
MIN_DRIVERS_PER_SESSION = 10
_TIME_CHANGE = re.compile(r"CAR (\d+)\b.*?\bTIME (\d+):(\d+\.\d+) (DELETED|REINSTATED)")

PRACTICE_DTYPES = {
    "season": "int64",
    "round": "int64",
    "session": "string",
    "session_key": "int64",
    "driver_number": "Int64",
    "driver_code": "string",
    "driver_id": "string",
    "lap_number": "Int64",
    "lap_start": "datetime64[ns, UTC]",
    "lap_time_s": "float64",
    "stint": "Int64",
    "compound": "string",
    "tyre_age": "Int64",
    "is_out_lap": "bool",
    "is_in_lap": "bool",
    "is_deleted": "bool",
    "is_green": "bool",
    "is_accurate": "bool",
}
PRACTICE_ONLY_DTYPES = {
    "season": "int64",
    "round": "int64",
    "session": "string",
    "driver_number": "Int64",
    "driver_code": "string",
}

# ---------------------------------------------------------------------------
# Race-control messages
# ---------------------------------------------------------------------------


def deleted_laps(
    race_control: Iterable[dict[str, Any]], laps: Iterable[dict[str, Any]]
) -> set[tuple[int, int]]:
    """Return (car number, lap number) for every lap whose time race control deleted.

    Messages read like "CAR 18 (STR) TIME 1:44.142 DELETED - TRACK LIMITS ... LAP 3". The
    "LAP 3" there counts laps differently from OpenF1 (it is one higher), so the lap is found
    by its time instead: the driver's lap with exactly that duration, the latest one that
    started before the message if there are several. A later "REINSTATED" undoes it.
    """
    by_driver: dict[int, list[tuple[pd.Timestamp, int, float]]] = defaultdict(list)
    for lap in laps:
        if lap.get("lap_duration") is not None and lap.get("date_start"):
            by_driver[int(lap["driver_number"])].append(
                (
                    pd.Timestamp(lap["date_start"]),
                    int(lap["lap_number"]),
                    float(lap["lap_duration"]),
                )
            )
    deleted: set[tuple[int, int]] = set()
    for message in sorted(race_control, key=lambda m: m.get("date") or ""):
        hit = _TIME_CHANGE.search(message.get("message") or "")
        if hit is None:
            continue
        car = int(hit.group(1))
        seconds = int(hit.group(2)) * 60 + float(hit.group(3))
        said_at = pd.Timestamp(message["date"])
        candidates = sorted(
            (start, number)
            for start, number, duration in by_driver[car]
            if abs(duration - seconds) <= TIME_MATCH_TOLERANCE_S
        )
        if not candidates:
            logger.debug("No lap of car %s matches deleted time %.3f", car, seconds)
            continue
        before = [number for start, number in candidates if start <= said_at]
        target = (car, before[-1] if before else candidates[0][1])
        if hit.group(4) == "DELETED":
            deleted.add(target)
        else:
            deleted.discard(target)
    return deleted


def non_green_windows(
    race_control: Iterable[dict[str, Any]], session_end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Time windows when the track was not fully green: yellow, red, VSC or safety car.

    A sector yellow lasts until that sector's CLEAR; a red flag until the next track-wide
    GREEN; a VSC or safety car until it ends. A track-wide GREEN, CLEAR or the chequered
    flag closes every open window. Anything still open closes at the session's scheduled
    end or the last message, whichever is later (sessions can overrun). Overlapping windows
    are merged, so two sectors going yellow together count once.
    """
    messages = sorted(race_control, key=lambda m: m.get("date") or "")
    opened: dict[str, pd.Timestamp] = {}
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def close(key: str, at: pd.Timestamp) -> None:
        """Close one open window at time ``at``."""
        start = opened.pop(key, None)
        if start is not None and at > start:
            windows.append((start, at))

    for message in messages:
        at = pd.Timestamp(message["date"])
        category = message.get("category") or ""
        flag = (message.get("flag") or "").upper()
        scope = (message.get("scope") or "").upper()
        text = (message.get("message") or "").upper()
        if category == "Flag":
            sector_key = f"sector-{message.get('sector')}" if scope == "SECTOR" else "track"
            if flag in ("YELLOW", "DOUBLE YELLOW"):
                opened.setdefault(sector_key, at)
            elif flag == "RED":
                opened.setdefault("red", at)
            elif flag == "CLEAR" and scope == "SECTOR":
                close(sector_key, at)
            elif (flag in ("GREEN", "CLEAR") and scope == "TRACK") or flag == "CHEQUERED":
                for key in list(opened):
                    close(key, at)
        elif category == "SafetyCar":
            if "DEPLOYED" in text:
                opened.setdefault("safety-car", at)
            elif "ENDING" in text or "IN THIS LAP" in text:
                close("safety-car", at)
    last_word = pd.Timestamp(messages[-1]["date"]) if messages else pd.Timestamp(session_end)
    for key in list(opened):
        close(key, max(pd.Timestamp(session_end), last_word))
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _is_green(
    start: pd.Timestamp | None, seconds: float, windows: list[tuple[float, float]]
) -> bool:
    """Whether a lap from ``start`` lasting ``seconds`` overlaps no non-green window.

    ``windows`` holds (start, end) pairs as seconds since 1970, which keeps the check to
    plain float comparisons.
    """
    if start is None:
        return False
    lap_start = start.timestamp()
    lap_end = lap_start + seconds
    return not any(w_start < lap_end and w_end > lap_start for w_start, w_end in windows)


# ---------------------------------------------------------------------------
# Laps
# ---------------------------------------------------------------------------


def lap_table(data: PracticeData) -> pd.DataFrame:
    """One row per timed lap of a practice session, with stint, tyre and lap-quality flags.

    Laps without a time (usually the first out-lap or a lap cut short by the end of the
    session) are left out; they cannot contribute a lap time to anything.
    """
    session = data.session
    codes = {int(d["driver_number"]): d.get("name_acronym") for d in data.drivers}
    stint_of: dict[tuple[int, int], tuple[int, str | None, int | None]] = {}
    in_laps: set[tuple[int, int]] = set()
    for stint in sorted(data.stints, key=lambda s: (s["driver_number"], s["stint_number"])):
        car, first = int(stint["driver_number"]), stint.get("lap_start")
        last = stint.get("lap_end")
        if first is None or last is None:
            continue
        age = stint.get("tyre_age_at_start")
        for lap_number in range(int(first), int(last) + 1):
            tyre_age = None if age is None else int(age) + lap_number - int(first)
            stint_of[(car, lap_number)] = (
                int(stint["stint_number"]),
                stint.get("compound"),
                tyre_age,
            )
        in_laps.add((car, int(last)))
    deleted = deleted_laps(data.race_control, data.laps)
    windows = [
        (start.timestamp(), end.timestamp())
        for start, end in non_green_windows(data.race_control, pd.Timestamp(session.end))
    ]

    rows = []
    for lap in data.laps:
        seconds = lap.get("lap_duration")
        if seconds is None:
            continue
        car, number = int(lap["driver_number"]), int(lap["lap_number"])
        start = pd.Timestamp(lap["date_start"]) if lap.get("date_start") else None
        stint, compound, tyre_age = stint_of.get((car, number), (None, None, None))
        sectors = [lap.get(f"duration_sector_{i}") for i in (1, 2, 3)]
        accurate = None not in sectors and abs(sum(sectors) - seconds) <= SECTOR_SUM_TOLERANCE_S
        rows.append(
            {
                "season": session.season,
                "round": session.round,
                "session": session.session,
                "session_key": session.session_key,
                "driver_number": car,
                "driver_code": codes.get(car),
                "driver_id": None,
                "lap_number": number,
                "lap_start": start,
                "lap_time_s": float(seconds),
                "stint": stint,
                "compound": compound,
                "tyre_age": tyre_age,
                "is_out_lap": bool(lap.get("is_pit_out_lap")),
                "is_in_lap": (car, number) in in_laps,
                "is_deleted": (car, number) in deleted,
                "is_green": _is_green(start, float(seconds), windows),
                "is_accurate": accurate,
            }
        )
    return typed_frame(rows, PRACTICE_DTYPES)


# ---------------------------------------------------------------------------
# A season
# ---------------------------------------------------------------------------


def build_practice_season(
    sessions: Iterable[PracticeData], entries: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build one season's practice laps and stand-in audit; return them plus warnings.

    ``entries`` is the season's Jolpica entry list (see ``mapping.round_entries``). A round
    without one yet (practice done, qualifying not) keeps empty driver ids and a warning;
    the next fetch after qualifying maps it.
    """
    frames, audit, warnings = [], [], []
    for data in sessions:
        session = data.session
        label = f"{session.season} round {session.round} {session.session}"
        laps = lap_table(data)
        if laps.empty:
            warnings.append(f"{label}: OpenF1 has no timed laps")
            continue
        round_entries = entries[entries["round"] == session.round]
        if round_entries.empty:
            warnings.append(f"{label}: no Jolpica entry list yet, drivers not mapped")
        else:
            drivers = laps[["driver_number", "driver_code"]].drop_duplicates()
            mapping, stand_ins = map_session_drivers(drivers, round_entries, context=label)
            laps["driver_id"] = laps["driver_number"].map(mapping).astype("string")
            audit += [
                {"season": session.season, "round": session.round, "session": session.session}
                | stand_in
                for stand_in in stand_ins
            ]
        drivers_with_laps = laps["driver_number"].nunique()
        if drivers_with_laps < MIN_DRIVERS_PER_SESSION:
            warnings.append(f"{label}: only {drivers_with_laps} drivers set a lap time")
        frames.append(laps)
    laps = pd.concat(frames, ignore_index=True) if frames else typed_frame([], PRACTICE_DTYPES)
    for warning in warnings:
        logger.warning(warning)
    return laps, typed_frame(audit, PRACTICE_ONLY_DTYPES), warnings
