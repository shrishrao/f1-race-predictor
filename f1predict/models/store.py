"""Save and load trained models: LightGBM's own text format plus a JSON description.

The ``.txt`` file holds the trees. The ``.json`` file next to it records how the model was
built: feature set and feature names, settings, the races it learned from, the simulation
temperature, and library versions. Loading checks that the two agree.

The DNF model that goes with a ranker is saved as ``dnf_<feature set>.json`` (plus a ``.txt``
when it's LightGBM): its kind, the starts and retirements per race it learned from (for the
field-wide DNF rate before the next race), and, for logistic regression, its fill values,
scaling and coefficients as numbers.

Both descriptions record what the model learned from (``trained_on``: races, rows, first and
last race), which is how ``predict`` tells whether saved models are up to date.

Nothing here uses pickle, which can run any code hidden in a file when it's loaded; these
files are only ever read as data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from f1predict import __version__, config
from f1predict.models.dnf import (
    DNF_KINDS,
    DNFModel,
    LightGBMDNF,
    LogisticDNF,
    make_dnf_model,
    starters,
)
from f1predict.models.ranker import RankerModel, training_rows

FORMAT_VERSION = 1
LOGISTIC_PARAMS = ("fill", "mean", "scale", "coef")


class ModelFileError(ValueError):
    """A saved model and its description don't match, or weren't made by this project."""


# ---------------------------------------------------------------------------
# What a model learned from
# ---------------------------------------------------------------------------


def race_label(season: int, round_number: int) -> str:
    """A race as named in saved descriptions and messages, e.g. ``2026 round 17``."""
    return f"{season} round {round_number}"


def training_summary(history: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """What the ranker and the DNF model learn from in ``history``, as saved with them.

    The ranker learns from finishers and the DNF model from starters, so each gets its own
    counts; both name the first and last race. ``history`` must hold at least one race.
    """
    races = history.drop_duplicates("race_seq").sort_values("race_seq")
    first, last = races.iloc[0], races.iloc[-1]
    span = {
        "first_race": race_label(int(first["season"]), int(first["round"])),
        "last_race": race_label(int(last["season"]), int(last["round"])),
    }
    finishers = training_rows(history)
    started = starters(history)
    ranker = {"races": int(finishers["race_seq"].nunique()), "finishers": len(finishers), **span}
    dnf = {
        "races": int(started["race_seq"].nunique()),
        "starters": len(started),
        "dnfs": int(started["status_class"].eq("dnf").sum()),
        **span,
    }
    return ranker, dnf


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def model_paths(name: str, folder: Path | None = None) -> tuple[Path, Path]:
    """The tree file and description file for a saved model, e.g. ``ranker_full.txt``."""
    root = folder or config.MODELS_DIR
    return root / f"{name}.txt", root / f"{name}.json"


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def save_ranker(
    model: RankerModel,
    name: str,
    trained_on: dict[str, Any],
    folder: Path | None = None,
    temperature: float | None = None,
) -> Path:
    """Write a trained ranker and its description; return the tree file's path.

    ``temperature`` is the simulation temperature to use with this ranker's scores.
    """
    if model.booster is None:
        raise RuntimeError("Only a trained model can be saved; call fit() first.")
    text_path, meta_path = model_paths(name, folder)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(text_path)
    description = {
        "format": FORMAT_VERSION,
        "f1predict_version": __version__,
        "lightgbm_version": lgb.__version__,
        "feature_set": model.feature_set,
        "features": model.features,
        "settings": model.settings(),
        "trained_on": trained_on,
        "temperature": temperature,
        "importance": model.importance().to_dict(orient="records"),
    }
    meta_path.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
    return text_path


def load_ranker(name: str, folder: Path | None = None) -> tuple[RankerModel, dict[str, Any]]:
    """Load a saved ranker and its description, checking that they belong together."""
    text_path, meta_path = model_paths(name, folder)
    if not text_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No saved model {name!r} in {text_path.parent}. Run `python -m f1predict train`."
        )
    description = json.loads(meta_path.read_text(encoding="utf-8"))
    if description.get("format") != FORMAT_VERSION:
        raise ModelFileError(f"{meta_path} is not a model description written by f1predict")
    booster = lgb.Booster(model_file=text_path)
    if booster.feature_name() != description["features"]:
        raise ModelFileError(f"{text_path} doesn't use the features its description lists")
    model = RankerModel(feature_set=description["feature_set"], **description["settings"])
    if model.features != description["features"]:
        raise ModelFileError(
            f"Feature set {model.feature_set!r} has changed since {name!r} was trained; "
            "run `python -m f1predict train` again."
        )
    model.booster = booster
    return model, description


# ---------------------------------------------------------------------------
# The DNF model
# ---------------------------------------------------------------------------


def save_dnf_model(
    model: DNFModel, name: str, trained_on: dict[str, Any], folder: Path | None = None
) -> Path:
    """Write a trained DNF model's description (and LightGBM's trees); return the JSON path.

    ``name`` is e.g. ``dnf_full``. A tree file left over from an earlier LightGBM choice is
    removed, so the folder never holds a ``.txt`` that doesn't belong to the ``.json``.
    """
    if model.counts is None:
        raise RuntimeError("Only a trained model can be saved; call fit() first.")
    text_path, meta_path = model_paths(name, folder)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    description: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "f1predict_version": __version__,
        "kind": model.kind,
        "window": model.window,
        "trained_on": trained_on,
        "race_counts": model.counts.to_dict(orient="list"),
    }
    if isinstance(model, LogisticDNF):
        description |= {"features": model.features, "params": model.params}
    if isinstance(model, LightGBMDNF):
        model.booster.save_model(text_path)
        description |= {
            "features": model.features,
            "lightgbm_version": lgb.__version__,
            "importance": model.importance().to_dict(orient="records"),
        }
    else:
        text_path.unlink(missing_ok=True)
    meta_path.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
    return meta_path


def load_dnf_model(name: str, folder: Path | None = None) -> tuple[DNFModel, dict[str, Any]]:
    """Load a saved DNF model and its description, checking that they belong together."""
    text_path, meta_path = model_paths(name, folder)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No saved DNF model {name!r} in {meta_path.parent}. Run `python -m f1predict train`."
        )
    description = json.loads(meta_path.read_text(encoding="utf-8"))
    if description.get("format") != FORMAT_VERSION or description.get("kind") not in DNF_KINDS:
        raise ModelFileError(f"{meta_path} is not a DNF model description written by f1predict")
    model = make_dnf_model(description["kind"], description["window"])
    model.counts = pd.DataFrame(description["race_counts"]).astype("int64")
    if isinstance(model, LogisticDNF):
        params = description["params"]
        if any(len(params[key]) != len(description["features"]) for key in LOGISTIC_PARAMS):
            raise ModelFileError(f"{meta_path}: the coefficients don't match the feature list")
        model.features, model.params = description["features"], params
    if isinstance(model, LightGBMDNF):
        if not text_path.exists():
            raise FileNotFoundError(
                f"{meta_path} describes LightGBM trees, but {text_path} is gone"
            )
        booster = lgb.Booster(model_file=text_path)
        if booster.feature_name() != description["features"]:
            raise ModelFileError(f"{text_path} doesn't use the features its description lists")
        model.features, model.booster = description["features"], booster
    return model, description
