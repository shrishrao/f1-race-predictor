"""Tests for saving and loading trained rankers (LightGBM text file + JSON description)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from synthetic import feature_rows, learnable_features

from f1predict.models.ranker import RankerModel
from f1predict.models.store import ModelFileError, load_ranker, model_paths, save_ranker

TRAINED_ON = {"races": 20, "finishers": 180, "first_race": "2020 round 1", "last_race": "x"}


@pytest.fixture(scope="module")
def frame():
    """The learnable championship from the ranker tests."""
    return learnable_features()


@pytest.fixture
def saved(frame, tmp_path):
    """A ranker trained on 2020 and saved to a temporary folder."""
    model = RankerModel(n_estimators=50).fit(frame[frame["season"] == 2020])
    save_ranker(model, "ranker_full", TRAINED_ON, tmp_path)
    return model, tmp_path


def test_a_loaded_model_scores_exactly_like_the_saved_one(saved, frame) -> None:
    """Same trees, same settings, same scores; the description says how it was built."""
    model, folder = saved
    loaded, description = load_ranker("ranker_full", folder)
    race = feature_rows(frame, 30)
    assert np.array_equal(model.predict(race)["score"], loaded.predict(race)["score"])
    assert loaded.settings() == model.settings()
    assert description["features"] == model.features
    assert description["trained_on"] == TRAINED_ON
    assert description["importance"][0]["feature"] == "quali_gap_pole_pct"


def test_a_missing_model_says_how_to_make_one(tmp_path) -> None:
    """No files yet: the error names the command that trains the model."""
    with pytest.raises(FileNotFoundError, match="f1predict train"):
        load_ranker("ranker_full", tmp_path)


def test_a_description_that_doesnt_match_is_refused(saved) -> None:
    """A tampered feature list, or a file this project didn't write, won't load."""
    _, folder = saved
    meta_path = model_paths("ranker_full", folder)[1]
    description = json.loads(meta_path.read_text(encoding="utf-8"))

    meta_path.write_text(json.dumps({**description, "features": ["grid"]}), encoding="utf-8")
    with pytest.raises(ModelFileError, match="features"):
        load_ranker("ranker_full", folder)

    meta_path.write_text(json.dumps({**description, "format": 99}), encoding="utf-8")
    with pytest.raises(ModelFileError, match="not a model description"):
        load_ranker("ranker_full", folder)


def test_an_untrained_model_cant_be_saved(tmp_path) -> None:
    """Saving before training is a mistake, not an empty file."""
    with pytest.raises(RuntimeError, match="fit"):
        save_ranker(RankerModel(), "ranker_full", TRAINED_ON, tmp_path)
