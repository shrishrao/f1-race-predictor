"""Write the files the Streamlit app reads into ``data/published/``.

The app never trains, predicts or downloads anything; it only reads what this module
writes. So the deployed app needs no LightGBM, and it can't show anything the command line
wouldn't:

- ``predictions/<season>_r<NN>.json``: one per race of the published seasons (2025 and
  2026) that can be predicted. A completed race is predicted by models trained only on the
  races before it, exactly as in the backtest; an upcoming race, once qualified, by the
  saved models. Each file holds the dict ``f1predict.predict`` returns.
- ``races.json``: the calendar of those seasons. Every race has its name, dates and status
  (``completed``, ``upcoming`` or ``scheduled``); a predicted race also names its file and
  favourite, and a completed one its winner and how the prediction scored next to the grid
  baseline in that race.
- ``metrics.json``: the backtest results behind the README: the full model against the
  grid baseline per test season with 95% confidence intervals, the ablations, the three
  DNF models, the calibration bins and what the models lean on.
- ``models/``: copies of the saved models the predictions came from.

Files are written in a fixed order with LF line endings and no timestamps, and only when
their content changes, so publishing again with nothing new changes no file (and git has
nothing to commit).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from f1predict import __version__, config
from f1predict.api import RaceData, load_race_data, predict_race, prediction_path
from f1predict.evaluate.metrics import AUC_METRIC, METRICS, clear_difference, compare, score_race
from f1predict.evaluate.report import (
    BASELINE,
    DNF_COMPARISON,
    FEATURE_LABELS,
    FULL,
    MODEL_METRICS,
    VARIANTS,
    Run,
    calibration_bins,
    load_description,
    load_dnf_comparison,
    load_run,
)
from f1predict.features.build import TARGET_COLUMNS
from f1predict.models import run_name
from f1predict.models.baseline import GridBaseline
from f1predict.models.dnf import DNF_KINDS, KIND_LABELS, SHORT_LABELS
from f1predict.models.store import model_paths, race_label
from f1predict.override import GridOverride, GridOverrideError

logger = logging.getLogger(__name__)

RACE_METRICS = ["winner_correct", "podium_hit_rate", "mae", "spearman"]
DNF_SUMMARY_METRICS = ["brier_dnf", "log_loss_dnf", "auc_dnf"]
DIGITS = 4

# ---------------------------------------------------------------------------
# Writing files
# ---------------------------------------------------------------------------


@dataclass
class PublishSummary:
    """What one ``publish`` run did: races by status, files changed, kept or removed, warnings."""

    folder: Path
    completed: int = 0
    upcoming: int = 0
    scheduled: int = 0
    changed: list[Path] = field(default_factory=list)
    unchanged: int = 0
    removed: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def to_json(value: Any) -> bytes:
    """JSON as published: indented UTF-8 with LF line endings, and never NaN."""
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _write(path: Path, content: bytes, summary: PublishSummary) -> None:
    """Write ``content`` unless the file already holds exactly that, and note which it was."""
    if path.exists() and path.read_bytes() == content:
        summary.unchanged += 1
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    summary.changed.append(path)


def _remove_others(folder: Path, keep: set[Path], summary: PublishSummary) -> None:
    """Delete the files in ``folder`` this run didn't write (e.g. a model file no longer used)."""
    if not folder.exists():
        return
    for path in sorted(folder.iterdir()):
        if path.is_file() and path not in keep:
            path.unlink()
            summary.removed.append(path)


def _num(value: Any) -> float | None:
    """A number rounded for publishing, or None when it's missing (JSON has no NaN)."""
    return None if pd.isna(value) else round(float(value), DIGITS)


def _text(value: Any) -> str | None:
    """A table cell as text, or None when it's missing."""
    return None if pd.isna(value) else str(value)


def _iso(value: Any) -> str | None:
    """A UTC timestamp as ISO 8601 text, e.g. ``2026-10-25T20:00:00Z``."""
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Races
# ---------------------------------------------------------------------------


def published_races(schedule: pd.DataFrame, seasons: tuple[int, ...]) -> pd.DataFrame:
    """The calendar rows to publish: every race of ``seasons`` that isn't left out."""
    calendar = schedule[schedule["season"].isin(list(seasons))]
    keys = zip(calendar["season"].astype(int), calendar["round"].astype(int), strict=True)
    kept = [key not in config.EXCLUDED_RACES for key in keys]
    return calendar[kept].sort_values(["season", "round"], ignore_index=True)


def race_scores(data: RaceData, result: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """How the prediction and the grid baseline did in one completed race.

    The backtest's order metrics: winner, podium hit rate, and MAE and Spearman over the
    cars that finished. The baseline learns from the same earlier races as the model.
    """
    season, round_number = result["race"]["season"], result["race"]["round"]
    features = data.features
    rows = features[(features["season"] == season) & (features["round"] == round_number)]
    known = features["status_class"].notna()
    history = features[(features["race_seq"] < rows["race_seq"].iloc[0]) & known]
    outcome = rows[["driver_id", *TARGET_COLUMNS]].assign(driver_id=rows["driver_id"].astype(str))
    model = pd.DataFrame(
        {
            "driver_id": [driver["driver_id"] for driver in result["drivers"]],
            "pred_rank": [driver["predicted_rank"] for driver in result["drivers"]],
        }
    )
    grid = GridBaseline().fit(history).predict(rows.drop(columns=TARGET_COLUMNS))
    baseline = pd.DataFrame(
        {"driver_id": grid["driver_id"].astype(str), "pred_rank": grid["pred_rank"]}
    )
    scores = {}
    for name, predicted in (("model", model), ("baseline", baseline)):
        scored = score_race(outcome.merge(predicted, on="driver_id", validate="one_to_one"))
        scores[name] = {metric: _num(scored[metric]) for metric in RACE_METRICS}
    return scores


def _driver(entry: dict[str, Any]) -> dict[str, Any]:
    """The fields of a prediction's driver entry that the race list needs."""
    return {
        "driver_id": entry["driver_id"],
        "code": entry["code"],
        "name": entry["name"],
        "predicted_rank": entry["predicted_rank"],
        "p_win": entry["p_win"],
    }


def race_entry(
    calendar_row: pd.Series,
    result: dict[str, Any] | None,
    scores: dict[str, dict[str, float | None]] | None,
) -> dict[str, Any]:
    """One race in ``races.json``: calendar details, then what was predicted and happened."""
    season, round_number = int(calendar_row["season"]), int(calendar_row["round"])
    entry: dict[str, Any] = {
        "season": season,
        "round": round_number,
        "name": _text(calendar_row["race_name"]),
        "circuit": _text(calendar_row["circuit_name"]),
        "country": _text(calendar_row["country"]),
        "qualifying_start": _iso(calendar_row["quali_start"]),
        "race_start": _iso(calendar_row["race_start"]),
        "status": "scheduled" if result is None else result["race"]["status"],
        "file": None,
    }
    if result is None:
        return entry
    drivers = result["drivers"]
    entry["file"] = f"predictions/{prediction_path(season, round_number).name}"
    entry["grid_override"] = result["grid_override"] is not None
    entry["favourite"] = _driver(max(drivers, key=lambda driver: driver["p_win"]))
    winner = next((d for d in drivers if d["actual"] and d["actual"]["position_text"] == "1"), None)
    if winner is not None:
        entry["winner"] = _driver(winner)
    if scores is not None:
        entry["scores"] = scores
    return entry


# ---------------------------------------------------------------------------
# Backtest results
# ---------------------------------------------------------------------------


def _interval(row: pd.Series, value: str = "value") -> dict[str, float | None]:
    """A value with its 95% confidence interval."""
    return {"value": _num(row[value]), "low": _num(row["low"]), "high": _num(row["high"])}


def _season_row(table: pd.DataFrame, season: int, metric: str) -> pd.Series:
    """The one row of a summary-like table for this season and metric."""
    return table[(table["season"] == season) & (table["metric"] == metric)].iloc[0]


def _gap(differences: pd.DataFrame, season: int, metric: str) -> dict[str, Any]:
    """A paired difference with its CI and verdict (+1 clearly better, -1 clearly worse)."""
    row = _season_row(differences, season, metric)
    better = METRICS[metric].better
    return {
        **_interval(row, "difference"),
        "verdict": clear_difference(better, row["low"], row["high"]),
    }


def _optional_run(name: str) -> Run | None:
    """A saved backtest, or None when it hasn't been run (only the ablations are optional)."""
    try:
        return load_run(name)
    except FileNotFoundError:
        logger.warning("No saved %s backtest; it's left out of metrics.json", name)
        return None


def _importance(ranker: dict[str, Any], dnf: dict[str, Any]) -> dict[str, Any]:
    """What the ranker's trees and the DNF model lean on, with readable input names."""
    gains = [
        {
            "feature": row["feature"],
            "label": FEATURE_LABELS.get(row["feature"], row["feature"]),
            "gain_share": _num(row["gain_share"]),
        }
        for row in ranker["importance"]
    ]
    if dnf["kind"] == "logistic":
        pairs = zip(dnf["features"], dnf["params"]["coef"], strict=True)
        weights = sorted(pairs, key=lambda pair: abs(pair[1]), reverse=True)
        unit = "coefficient per standard deviation"
    elif dnf["kind"] == "lightgbm":
        weights = [(row["feature"], row["gain_share"]) for row in dnf["importance"]]
        unit = "share of gain"
    else:
        weights, unit = [], None
    inputs = [
        {"feature": name, "label": FEATURE_LABELS.get(name, name), "value": _num(value)}
        for name, value in weights
    ]
    return {"ranker": gains, "dnf": {"kind": dnf["kind"], "unit": unit, "inputs": inputs}}


def backtest_metrics(seasons: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Everything the app's performance page shows, from the saved backtests and models.

    ``seasons`` defaults to the test seasons. Needs what ``report`` needs: the baseline and
    full-model backtests, the DNF comparison and the trained models. Ablations that haven't
    been run are left out.
    """
    seasons = seasons or config.TEST_SEASONS
    baseline, full = load_run(BASELINE), load_run(FULL)
    test = [season for season in seasons if season in full.seasons]
    dnf_summary, _ = load_dnf_comparison()
    ranker, dnf = load_description(FULL), load_description(DNF_COMPARISON)
    available = {FULL: full, BASELINE: baseline}
    for name in VARIANTS:
        if name not in available:
            available[name] = _optional_run(name)
    runs = {name: available[name] for name in VARIANTS if available[name] is not None}

    gaps = compare(full.races, baseline.races)
    vs_baseline = [
        {
            "metric": metric,
            "season": season,
            "model": _interval(_season_row(full.summary, season, metric)),
            "baseline": _interval(_season_row(baseline.summary, season, metric)),
            "difference": _gap(gaps, season, metric),
        }
        for metric in MODEL_METRICS
        for season in test
    ]
    ablations = []
    for name, run in runs.items():
        variant_gaps = None if name == FULL else compare(run.races, full.races)
        for metric in MODEL_METRICS:
            for season in test:
                row: dict[str, Any] = {
                    "variant": name,
                    "label": VARIANTS[name],
                    "metric": metric,
                    "season": season,
                    "value": _num(run.value(season, metric)),
                }
                if variant_gaps is not None:
                    row["difference"] = _gap(variant_gaps, season, metric)
                ablations.append(row)
    dnf_rows = [
        {
            "model": kind,
            "label": SHORT_LABELS[kind],
            "metric": metric,
            "season": season,
            **_interval(_season_row(dnf_summary[dnf_summary["model"] == kind], season, metric)),
        }
        for metric in DNF_SUMMARY_METRICS
        for kind in DNF_KINDS
        for season in test
    ]
    calibration = []
    for name, run in (("model", full), ("baseline", baseline)):
        calibration += [
            {
                "model": name,
                "outcome": row.outcome,
                "bin_low": _num(row.bin_low),
                "bin_high": _num(row.bin_high),
                "n": int(row.n),
                "predicted": _num(row.predicted),
                "observed": _num(row.observed),
                "low": _num(row.low),
                "high": _num(row.high),
            }
            for row in calibration_bins(run.predictions, test).itertuples()
        ]
    specs = METRICS | {"auc_dnf": AUC_METRIC}
    return {
        "f1predict_version": __version__,
        "validation_season": config.VALIDATION_SEASON,
        "test_seasons": test,
        "races": {str(season): full.race_count(season) for season in test},
        "metrics": {
            name: {"label": spec.label, "better": spec.better, "format": spec.fmt}
            for name, spec in specs.items()
        },
        "vs_baseline": vs_baseline,
        "ablations": ablations,
        "dnf_models": {"chosen": dnf["kind"], "label": KIND_LABELS[dnf["kind"]], "rows": dnf_rows},
        "calibration": calibration,
        "importance": _importance(ranker, dnf),
    }


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def _stale_backtests(entries: list[dict[str, Any]], metrics: dict[str, Any]) -> list[str]:
    """Warnings for seasons with more races run than the saved backtests have scored."""
    warnings = []
    for season in metrics["test_seasons"]:
        run = sum(1 for e in entries if e["season"] == season and e["status"] == "completed")
        scored = metrics["races"][str(season)]
        if run > scored:
            warnings.append(
                f"The backtests have scored {scored} of the {run} races run in {season}, so the "
                "app's season numbers are behind. Run `python -m f1predict backtest` and "
                "`python -m f1predict backtest --model ranker` (and `report`) to catch up."
            )
    return warnings


def model_files() -> list[Path]:
    """The saved model files behind the published predictions (trees and descriptions)."""
    files = []
    for name in (run_name("ranker", "full"), run_name("dnf", "full")):
        files += [path for path in model_paths(name) if path.exists()]
    return files


def publish(
    seasons: tuple[int, ...] | None = None,
    grid_override: GridOverride | None = None,
    folder: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> PublishSummary:
    """Write the predictions, the race list, the backtest results and the models for the app.

    ``seasons`` defaults to the test seasons. ``grid_override`` applies to the upcoming race,
    and is refused if there isn't one. ``progress``, if given, is called with a label like
    ``"2026 round 3"`` after each race.
    """
    seasons = seasons or config.TEST_SEASONS
    root = folder or config.PUBLISHED_DIR
    summary = PublishSummary(root)
    metrics = backtest_metrics(seasons)
    data = load_race_data()
    features = data.features
    written: set[Path] = set()
    entries = []
    for calendar_row in published_races(data.schedule, seasons).itertuples(index=False):
        row = pd.Series(calendar_row._asdict())
        season, round_number = int(row["season"]), int(row["round"])
        race = features[(features["season"] == season) & (features["round"] == round_number)]
        result = scores = None
        if race.empty:
            summary.scheduled += 1
        else:
            completed = bool(race["status_class"].notna().all())
            override = None if completed else grid_override
            result = predict_race(data, season, round_number, grid_override=override)
            path = prediction_path(season, round_number, folder=root / "predictions")
            _write(path, to_json(result), summary)
            written.add(path)
            if completed:
                summary.completed += 1
                scores = race_scores(data, result)
            else:
                summary.upcoming += 1
        entries.append(race_entry(row, result, scores))
        if progress is not None:
            progress(race_label(season, round_number))
    if grid_override is not None and not summary.upcoming:
        raise GridOverrideError("There's no upcoming qualified race to apply the override to.")
    _remove_others(root / "predictions", written, summary)
    summary.warnings += _stale_backtests(entries, metrics)

    known = features[features["status_class"].notna()].sort_values("race_seq")
    last = known.iloc[-1]
    index = {
        "f1predict_version": __version__,
        "seasons": list(seasons),
        "data_through": race_label(int(last["season"]), int(last["round"])),
        "races": entries,
    }
    _write(root / "races.json", to_json(index), summary)
    _write(root / "metrics.json", to_json(metrics), summary)

    # Line endings are normalized to LF, like every committed text file: on Windows the
    # descriptions and LightGBM's last two lines are written with CRLF.
    copies = set()
    for source in model_files():
        target = root / "models" / source.name
        _write(target, source.read_bytes().replace(b"\r\n", b"\n"), summary)
        copies.add(target)
    _remove_others(root / "models", copies, summary)
    return summary
