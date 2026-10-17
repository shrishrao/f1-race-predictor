"""Match practice drivers (car number and 3-letter code) to Jolpica driver ids, round by round.

A round's entry list comes from Jolpica: everyone in its qualifying, sprint or race
results, with the car number they used that weekend. Each practice driver is matched on
their code first and their car number second. Two outcomes are deliberately not silent:

- A driver who matches nobody is an FP1 stand-in (a rookie or reserve who never races that
  weekend). They are returned in a "practice-only" list for an audit table, not dropped.
- A real contradiction (the code points to one driver and the number to another, or two
  practice drivers claim the same entrant) raises ``DriverMappingError``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

ENTRY_COLUMNS = ["season", "round", "driver_id", "driver_code", "car_number"]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DriverMappingError(ValueError):
    """A practice driver could not be matched to an entrant consistently."""


class UnknownDriverError(DriverMappingError):
    """A practice driver matches nobody in the round's entry list (usually an FP1 stand-in)."""


# ---------------------------------------------------------------------------
# Entry lists
# ---------------------------------------------------------------------------


def round_entries(*tables: pd.DataFrame) -> pd.DataFrame:
    """Everyone entered in each round: the union of results, qualifying and sprint rows.

    Each table needs the ``ENTRY_COLUMNS``; the result has one row per driver per round.
    """
    frames = [table[ENTRY_COLUMNS] for table in tables if not table.empty]
    if not frames:
        return pd.DataFrame(columns=ENTRY_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["season", "round", "driver_id"], ignore_index=True)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def map_driver(code: str | None, number: int | None, entries: pd.DataFrame) -> str:
    """Return the Jolpica driver id for one practice driver in one round.

    ``entries`` is that round's entry list. Raises ``UnknownDriverError`` if neither the
    code nor the car number matches anyone, and ``DriverMappingError`` if they point to
    different drivers or match more than one.
    """
    by_code = entries.loc[entries["driver_code"].eq(code).fillna(False), "driver_id"]
    by_number = entries.loc[entries["car_number"].eq(number).fillna(False), "driver_id"]
    if code is None:
        by_code = by_code.iloc[0:0]
    if number is None:
        by_number = by_number.iloc[0:0]
    for label, value, hits in (("code", code, by_code), ("car number", number, by_number)):
        if hits.nunique() > 1:
            raise DriverMappingError(f"{label} {value} matches several entrants: {sorted(hits)}")
    from_code = str(by_code.iloc[0]) if len(by_code) else None
    from_number = str(by_number.iloc[0]) if len(by_number) else None
    if from_code and from_number and from_code != from_number:
        raise DriverMappingError(
            f"code {code} points to {from_code} but car number {number} points to {from_number}"
        )
    if from_code or from_number:
        return from_code or from_number
    raise UnknownDriverError(f"{code} #{number} is not in this round's entry list")


def map_session_drivers(
    drivers: pd.DataFrame, entries: pd.DataFrame, *, context: str = ""
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Map every driver of one practice session; return ``{car number: driver id}`` and stand-ins.

    ``drivers`` has one row per practice driver with ``driver_number`` and ``driver_code``.
    Stand-ins come back in the second list; contradictions raise ``DriverMappingError``
    with ``context`` (for example "2024 round 5 FP1") in the message.
    """
    mapping: dict[int, str] = {}
    practice_only: list[dict[str, Any]] = []
    for row in drivers.itertuples(index=False):
        code = None if pd.isna(row.driver_code) else str(row.driver_code)
        number = int(row.driver_number)
        try:
            mapping[number] = map_driver(code, number, entries)
        except UnknownDriverError:
            practice_only.append({"driver_number": number, "driver_code": code})
        except DriverMappingError as exc:
            raise DriverMappingError(f"{context}: {exc}") from exc
    claimed = Counter(mapping.values())
    repeated = sorted(driver for driver, count in claimed.items() if count > 1)
    if repeated:
        raise DriverMappingError(f"{context}: several practice drivers map to {repeated}")
    return mapping, practice_only
