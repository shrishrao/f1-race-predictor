"""Save and load trained rankers: LightGBM's own text format plus a JSON description.

The ``.txt`` file holds the trees. The ``.json`` file next to it records how the model was
built: feature set and feature names, settings, the races it learned from, and library
versions. Loading checks that the two agree. Nothing here uses pickle, which can run any
code hidden in a file when it's loaded; these files are only ever read as data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb

from f1predict import __version__, config
from f1predict.models.ranker import RankerModel

FORMAT_VERSION = 1


class ModelFileError(ValueError):
    """A saved model and its description don't match, or weren't made by this project."""


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
    model: RankerModel, name: str, trained_on: dict[str, Any], folder: Path | None = None
) -> Path:
    """Write a trained ranker and its description; return the tree file's path."""
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
