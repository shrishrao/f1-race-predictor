"""Grid overrides: the starting grid when it isn't the qualifying order.

After qualifying the grid can still change. A driver who takes a new gearbox or engine
drops some places, and a car rebuilt under parc fermé starts from the pit lane. The
stewards publish the final grid before the race, and an override file passes it on:

    {"season": 2026, "round": 18, "grid": {"hamilton": 15}, "pitlane": ["stroll"]}

- ``grid`` puts drivers (by Jolpica driver id) in a starting slot, 1 being pole. Everyone
  else keeps their order and fills the free slots: a driver dropped from 3rd to 15th moves
  the cars from 4th to 15th up one place each.
- ``pitlane`` lists cars starting from the pit lane, behind the whole grid.
- ``season`` and ``round`` are optional. When given, they must match the race being
  predicted, so a file written for one race can't quietly be used for another.

The file is checked twice. ``GridOverride`` (a pydantic model) checks its shape: only
these keys, whole-number slots from 1, no slot or driver named twice. Then
``apply_grid_override`` checks it against the race: every driver must be entered, and
every slot must exist on that race's grid.
"""

from __future__ import annotations

import codecs
import difflib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from f1predict.models import require_one_race

Slot = Annotated[int, Field(strict=True, ge=1)]
DriverId = Annotated[str, Field(strict=True, min_length=1)]

# ---------------------------------------------------------------------------
# The file format
# ---------------------------------------------------------------------------


class GridOverrideError(ValueError):
    """A grid override is malformed, or doesn't fit the race it's applied to."""


class GridOverride(BaseModel):
    """The contents of a grid override file, checked for shape but not yet against a race.

    Unknown keys are refused, so a typo like ``"pitlnae"`` is an error rather than an
    override that silently changes nothing. Slots must be whole numbers from 1: ``"3"``,
    ``3.0`` and ``true`` are refused too.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    season: int | None = Field(default=None, strict=True)
    round: int | None = Field(default=None, strict=True)
    grid: dict[DriverId, Slot] = Field(default_factory=dict)
    pitlane: list[DriverId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _each_slot_and_driver_once(self) -> GridOverride:
        """Refuse an empty override, a slot given twice, or a driver named twice."""
        if not self.grid and not self.pitlane:
            raise ValueError("the override is empty: give 'grid' slots, 'pitlane' drivers or both")
        shared = sorted(slot for slot, count in Counter(self.grid.values()).items() if count > 1)
        if shared:
            raise ValueError(f"more than one driver is given slot {_listing(shared)}")
        twice = sorted(driver for driver, count in Counter(self.pitlane).items() if count > 1)
        if twice:
            raise ValueError(f"'pitlane' names {_listing(twice)} more than once")
        both = sorted(set(self.grid) & set(self.pitlane))
        if both:
            raise ValueError(f"{_listing(both)} can't have a grid slot and start from the pit lane")
        return self


def _listing(values: Iterable[Any]) -> str:
    """Values joined for a message, e.g. ``'norris', 'piastri'`` or ``3, 7``."""
    return ", ".join(repr(value) if isinstance(value, str) else str(value) for value in values)


def _describe(error: ValidationError) -> str:
    """Pydantic's errors as readable text, one ``where: what`` part per problem."""
    parts = []
    for item in error.errors(include_url=False):
        where = ".".join(str(part) for part in item["loc"])
        message = item["msg"].removeprefix("Value error, ")
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def parse_grid_override(data: GridOverride | Mapping[str, Any]) -> GridOverride:
    """Check an override given as a dict (e.g. a parsed JSON body); a GridOverride passes."""
    if isinstance(data, GridOverride):
        return data
    try:
        return GridOverride.model_validate(data)
    except ValidationError as exc:
        raise GridOverrideError(f"Invalid grid override: {_describe(exc)}") from exc


def load_grid_override(path: Path | str) -> GridOverride:
    """Read and check an override file.

    UTF-8 (with or without a byte-order mark) and UTF-16 are both accepted, because Windows
    PowerShell's ``>`` and ``Out-File`` write UTF-16 by default.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GridOverrideError(f"Can't read {path}: {exc.strerror or exc}") from exc
    utf16 = raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
    try:
        text = raw.decode("utf-16" if utf16 else "utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GridOverrideError(f"{path} isn't UTF-8 or UTF-16 text") from exc
    try:
        return GridOverride.model_validate_json(text)
    except ValidationError as exc:
        raise GridOverrideError(f"{path}: {_describe(exc)}") from exc


# ---------------------------------------------------------------------------
# Applying to a race
# ---------------------------------------------------------------------------


def _suggestion(name: str, race: pd.DataFrame) -> str:
    """`` (did you mean 'norris'?)`` for a driver code or a near miss, else nothing."""
    ids = [str(driver) for driver in race["driver_id"]]
    codes = {
        str(code).upper(): driver
        for code, driver in zip(race["driver_code"], ids, strict=True)
        if pd.notna(code)
    }
    guess = codes.get(name.upper()) or next(iter(difflib.get_close_matches(name, ids, n=1)), None)
    return f" (did you mean {guess!r}?)" if guess else ""


def apply_grid_override(race: pd.DataFrame, override: GridOverride) -> pd.DataFrame:
    """One race's feature rows with the override applied to ``grid`` and ``pitlane_start``.

    Drivers the override doesn't name keep their order (qualifying for a race not run yet,
    the actual grid for one that was) and fill the free slots; a car already starting from
    the pit lane stays there unless it's given a slot. As in ``grid_features``, pit-lane
    starters get ``pitlane_start = 1`` and a ``grid`` one slot behind the last car.

    The rows come back in the new starting order (grid, then driver id), the order the
    features table keeps every race in. The simulation draws its random numbers car by car
    in row order, so this makes an overridden race simulate exactly as it would if the new
    grid had been in the data all along.
    """
    require_one_race(race)
    season, round_number = (int(value) for value in race[["season", "round"]].iloc[0])
    label = f"{season} round {round_number}"
    for key, given, actual in (
        ("season", override.season, season),
        ("round", override.round, round_number),
    ):
        if given is not None and given != actual:
            raise GridOverrideError(f"The override is for {key} {given}, but the race is {label}.")

    entered = [str(driver) for driver in race["driver_id"]]
    unknown = [driver for driver in [*override.grid, *override.pitlane] if driver not in entered]
    if unknown:
        named = ", ".join(f"{driver!r}{_suggestion(driver, race)}" for driver in unknown)
        raise GridOverrideError(
            f"Not entered in {label}: {named}. Entered: {', '.join(sorted(entered))}."
        )

    already = {str(d) for d in race.loc[race["pitlane_start"].eq(1), "driver_id"]}
    in_pitlane = set(override.pitlane) | (already - set(override.grid))
    on_grid = race[~race["driver_id"].isin(in_pitlane)]
    cars = len(on_grid)
    beyond = {driver: slot for driver, slot in override.grid.items() if slot > cars}
    if beyond:
        named = ", ".join(f"{driver!r} to {slot}" for driver, slot in beyond.items())
        raise GridOverrideError(
            f"{label} has {cars} grid slots (with {len(in_pitlane)} in the pit lane), "
            f"so the override can't move {named}."
        )

    taken = set(override.grid.values())
    free = iter(slot for slot in range(1, cars + 1) if slot not in taken)
    others = on_grid[~on_grid["driver_id"].isin(list(override.grid))]
    others = others.sort_values(["grid", "driver_id"], kind="stable")
    slots = {**override.grid, **{str(driver): next(free) for driver in others["driver_id"]}}
    out = race.copy()
    out["grid"] = [
        float(cars + 1) if str(driver) in in_pitlane else float(slots[str(driver)])
        for driver in out["driver_id"]
    ]
    out["pitlane_start"] = [float(str(driver) in in_pitlane) for driver in out["driver_id"]]
    return out.sort_values(["grid", "driver_id"], kind="stable")
