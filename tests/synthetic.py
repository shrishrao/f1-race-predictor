"""Small, seeded, made-up data in the exact shape of the real tables.

``make_tables`` builds a championship in the shape of the interim tables. The leakage test
needs many races with results, qualifying, sprints and practice, and the real data isn't in
the repo (or CI). It has two seasons of five rounds each: three teams, six drivers, one
sprint weekend per season, a pit-lane start, retirements, a non-starter, and an FP1
stand-in, all from a fixed random seed.

``learnable_features`` builds a features table with one obvious signal, for checking that
the ranker actually learns. ``dnf_features`` does the same for the DNF models: cars that
start further back retire more often.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from f1predict.data.practice import PRACTICE_DTYPES
from f1predict.data.status import classify_status
from f1predict.data.tables import (
    QUALIFYING_DTYPES,
    RESULT_DTYPES,
    SCHEDULE_DTYPES,
    typed_frame,
)
from f1predict.features.build import FEATURE_COLUMNS, TARGET_COLUMNS, RaceTables

DRIVERS = [
    ("hamilton", "HAM", 44, "mercedes"),
    ("russell", "RUS", 63, "mercedes"),
    ("leclerc", "LEC", 16, "ferrari"),
    ("sainz", "SAI", 55, "ferrari"),
    ("bottas", "BOT", 77, "alfa"),
    ("zhou", "ZHO", 24, "alfa"),
]
CIRCUITS = ["bahrain", "jeddah", "monaco"]
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def _team(constructor: str, season: int) -> str:
    """Alfa Romeo becomes Sauber in the second season (same lineage, new constructor id)."""
    return "sauber" if constructor == "alfa" and season > 2023 else constructor


def make_tables(
    seasons: tuple[int, ...] = (2023, 2024), rounds: int = 5, seed: int = 7
) -> RaceTables:
    """Build the five interim tables for a made-up championship."""
    rng = np.random.default_rng(seed)
    schedule, results, qualifying, sprint, practice = [], [], [], [], []
    for season in seasons:
        for rnd in range(1, rounds + 1):
            race_day = datetime(season, 3, 1, 15, tzinfo=UTC) + timedelta(weeks=2 * rnd)
            sprint_weekend = rnd == 3
            schedule.append(
                {
                    "season": season,
                    "round": rnd,
                    "race_name": f"Race {rnd}",
                    "circuit_id": CIRCUITS[(rnd - 1) % len(CIRCUITS)],
                    "circuit_name": "x",
                    "locality": "x",
                    "country": "x",
                    "fp1_start": race_day - timedelta(days=2),
                    "fp2_start": None if sprint_weekend else race_day - timedelta(days=2, hours=-4),
                    "fp3_start": None if sprint_weekend else race_day - timedelta(days=1, hours=3),
                    "sprint_quali_start": race_day - timedelta(days=2, hours=-4)
                    if sprint_weekend
                    else None,
                    "sprint_start": race_day - timedelta(days=1, hours=4)
                    if sprint_weekend
                    else None,
                    "quali_start": race_day - timedelta(days=1),
                    "race_start": race_day,
                    "is_sprint_weekend": sprint_weekend,
                }
            )
            pace = rng.normal(0, 0.4, len(DRIVERS)) + np.arange(len(DRIVERS)) * 0.15
            grid = np.argsort(np.argsort(pace + rng.normal(0, 0.2, len(DRIVERS)))) + 1
            order = np.argsort(pace + rng.normal(0, 0.6, len(DRIVERS)))
            statuses = ["Finished"] * len(DRIVERS)
            statuses[order[-1]] = "Retired" if rnd % 2 else "Lapped"
            if (season, rnd) == (seasons[-1], 2):
                statuses[order[-2]] = "Did not start"
            for position, idx in enumerate(order, start=1):
                driver_id, code, number, constructor = DRIVERS[idx]
                status = statuses[idx]
                classified = classify_status(status).value == "finished"
                results.append(
                    {
                        "season": season,
                        "round": rnd,
                        "driver_id": driver_id,
                        "driver_code": code,
                        "driver_name": driver_id,
                        "car_number": number,
                        "constructor_id": _team(constructor, season),
                        "constructor_name": constructor,
                        "grid": 0 if (season, rnd, idx) == (seasons[0], 4, 0) else int(grid[idx]),
                        "position": position,
                        "position_text": str(position) if classified else "R",
                        "points": float(POINTS.get(position, 0)) if classified else 0.0,
                        "laps": 50 if classified else 20,
                        "status": status,
                        "status_class": classify_status(status).value,
                        "time_ms": None,
                    }
                )
            for slot, idx in enumerate(np.argsort(pace), start=1):
                driver_id, code, number, constructor = DRIVERS[idx]
                base = 90 + pace[idx]
                qualifying.append(
                    {
                        "season": season,
                        "round": rnd,
                        "driver_id": driver_id,
                        "driver_code": code,
                        "car_number": number,
                        "constructor_id": _team(constructor, season),
                        "position": slot,
                        "q1_s": base + 0.4,
                        "q2_s": base + 0.2 if slot <= 4 else np.nan,
                        "q3_s": base if slot <= 2 else np.nan,
                    }
                )
            if sprint_weekend:
                for position, idx in enumerate(
                    np.argsort(pace + rng.normal(0, 0.3, len(DRIVERS))), 1
                ):
                    driver_id, code, number, constructor = DRIVERS[idx]
                    sprint.append(
                        {
                            "season": season,
                            "round": rnd,
                            "driver_id": driver_id,
                            "driver_code": code,
                            "driver_name": driver_id,
                            "car_number": number,
                            "constructor_id": _team(constructor, season),
                            "constructor_name": constructor,
                            "grid": position,
                            "position": position,
                            "position_text": str(position),
                            "points": 0.0,
                            "laps": 20,
                            "status": "Finished",
                            "status_class": "finished",
                            "time_ms": None,
                        }
                    )
            sessions = ["FP1"] if sprint_weekend else ["FP1", "FP2", "FP3"]
            for session in sessions:
                start = schedule[-1][f"{session.lower()}_start"]
                cars = list(enumerate(DRIVERS))
                if session == "FP1" and rnd == 2:
                    cars.append((len(DRIVERS), ("", "ROO", 99, "mercedes")))
                for idx, (driver_id, code, number, _) in cars:
                    base = 91 + (pace[idx] if idx < len(DRIVERS) else -0.3)
                    lap_start = start
                    for stint, compound in ((1, "MEDIUM"), (2, "SOFT")):
                        for lap in range(1, 9):
                            seconds = base + rng.normal(0, 0.15) + (25 if lap in (1, 8) else 0)
                            lap_number = (stint - 1) * 8 + lap
                            practice.append(
                                {
                                    "season": season,
                                    "round": rnd,
                                    "session": session,
                                    "session_key": season * 1000
                                    + rnd * 10
                                    + sessions.index(session),
                                    "driver_number": number,
                                    "driver_code": code,
                                    "driver_id": driver_id or None,
                                    "lap_number": lap_number,
                                    "lap_start": lap_start,
                                    "lap_time_s": seconds,
                                    "stint": stint,
                                    "compound": compound,
                                    "tyre_age": lap - 1,
                                    "is_out_lap": lap == 1,
                                    "is_in_lap": lap == 8,
                                    "is_deleted": bool(rng.random() < 0.03),
                                    "is_green": bool(rng.random() > 0.05),
                                    "is_accurate": True,
                                }
                            )
                            lap_start = lap_start + timedelta(seconds=float(seconds))
    return RaceTables(
        schedule=typed_frame(schedule, SCHEDULE_DTYPES),
        results=typed_frame(results, RESULT_DTYPES),
        qualifying=typed_frame(qualifying, QUALIFYING_DTYPES),
        sprint=typed_frame(sprint, RESULT_DTYPES),
        practice=typed_frame(practice, PRACTICE_DTYPES),
    )


def scramble_after(
    tables: RaceTables, season: int, round_number: int, seed: int = 99
) -> RaceTables:
    """Shuffle the outcomes of every race after the given one (positions and statuses).

    Features for the given race must not change: a feature that peeks into the future would.
    """
    rng = np.random.default_rng(seed)
    results = tables.results.copy()
    later = (results["season"] > season) | (
        (results["season"] == season) & (results["round"] > round_number)
    )
    for (_, _), race in results[later].groupby(["season", "round"]):
        results.loc[race.index, "position"] = rng.permutation(race["position"].to_numpy())
        results.loc[race.index, "status_class"] = rng.choice(["finished", "dnf"], len(race))
    return RaceTables(
        schedule=tables.schedule,
        results=results,
        qualifying=tables.qualifying,
        sprint=tables.sprint,
        practice=tables.practice,
    )


def learnable_features(races: int = 40, cars: int = 10, seed: int = 0) -> pd.DataFrame:
    """A features table (20 races a season from 2020) where the smallest quali gap always wins.

    Every other feature is missing, the grid is a random shuffle, and one random car per
    race retires: it's classified last and has no finishing rank.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for seq in range(races):
        gaps = rng.uniform(0, 3, cars)
        grid = rng.permutation(cars) + 1
        retiree = int(rng.integers(cars))
        order = [int(car) for car in np.argsort(gaps) if car != retiree]
        position = {car: place for place, car in enumerate(order, start=1)} | {retiree: cars}
        for car in range(cars):
            finished = car != retiree
            rows.append(
                {
                    "season": 2020 + seq // 20,
                    "round": seq % 20 + 1,
                    "race_seq": seq,
                    "circuit_id": "monza",
                    "driver_id": f"d{car}",
                    "driver_code": f"D{car}",
                    "team_id": f"t{car // 2}",
                    "grid": float(grid[car]),
                    "pitlane_start": 0.0,
                    "quali_gap_pole_pct": gaps[car],
                    "position": position[car],
                    "status_class": "finished" if finished else "dnf",
                    "finished": finished,
                    "finish_rank": float(position[car]) if finished else np.nan,
                    "n_finishers": float(cars - 1),
                }
            )
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame


def dnf_features(
    races: int = 60, cars: int = 20, seed: int = 0, jump_at: int | None = None
) -> pd.DataFrame:
    """A features table (20 races a season from 2020) where the back of the grid retires more.

    A car's DNF chance rises in a straight line from 5% on pole to 40% in last place; from
    race ``jump_at`` on (if given) every chance doubles, like a regulation reset. The grid
    is a random shuffle and the other DNF inputs are pure noise. Finishers finish in grid
    order. ``true_p_dnf`` holds the chance each car was given, the best any model can do.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for seq in range(races):
        grid = rng.permutation(cars) + 1
        chance = 0.05 + 0.35 * (grid - 1) / (cars - 1)
        if jump_at is not None and seq >= jump_at:
            chance = 2 * chance
        retired = rng.random(cars) < chance
        finishers = [int(car) for car in np.argsort(grid) if not retired[car]]
        rank = {car: place for place, car in enumerate(finishers, start=1)}
        for car in range(cars):
            rows.append(
                {
                    "season": 2020 + seq // 20,
                    "round": seq % 20 + 1,
                    "race_seq": seq,
                    "circuit_id": "monza",
                    "driver_id": f"d{car}",
                    "driver_code": f"D{car}",
                    "team_id": f"t{car // 2}",
                    "grid": float(grid[car]),
                    "pitlane_start": 0.0,
                    "quali_gap_pole_pct": rng.uniform(0, 3),
                    "drv_dnf_rate": rng.uniform(0.05, 0.3),
                    "team_dnf_rate": rng.uniform(0.05, 0.3),
                    "circuit_dnf_rate": rng.uniform(0.05, 0.3),
                    "true_p_dnf": chance[car],
                    "position": rank.get(car, len(finishers) + 1),
                    "status_class": "dnf" if retired[car] else "finished",
                    "finished": not retired[car],
                    "finish_rank": float(rank[car]) if car in rank else np.nan,
                    "n_finishers": float(len(finishers)),
                }
            )
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame


def feature_rows(features: pd.DataFrame, race_seq: int) -> pd.DataFrame:
    """One race's rows without the outcome, as the backtest passes them to ``predict``."""
    return features[features["race_seq"] == race_seq].drop(columns=TARGET_COLUMNS)
