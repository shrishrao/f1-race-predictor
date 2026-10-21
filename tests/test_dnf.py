"""Tests for the DNF models, their scoring, the choice on 2024, and saving them.

Most tests use a made-up championship where the DNF chance rises from 5% on pole to 40% in
last place, and everything else is noise. A model that can't find that is broken; a model
that sees the base rate move only after the races that moved it is leakage-safe.
"""

from __future__ import annotations

import json
from functools import partial

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from synthetic import dnf_features, feature_rows

from f1predict import config
from f1predict.evaluate import tuning
from f1predict.evaluate.backtest import PredictionError, check_prediction, predict_races
from f1predict.evaluate.dnf import compare_dnf_models, save_dnf_comparison
from f1predict.evaluate.metrics import (
    calibration_table,
    dnf_race_metrics,
    dnf_scores,
    roc_auc,
    season_auc,
)
from f1predict.models.dnf import (
    DNF_FEATURES,
    DNF_KINDS,
    FIELD_RATE,
    BaseRateDNF,
    LightGBMDNF,
    LogisticDNF,
    make_dnf_model,
    race_counts,
    rate_before,
    rates_before,
    with_field_rate,
)
from f1predict.models.store import ModelFileError, load_dnf_model, model_paths, save_dnf_model

KEYS = ["season", "round", "driver_id"]
LEARNED = [LogisticDNF, LightGBMDNF]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """The made-up championship: 2020-2022, 20 races each, back of the grid retires more."""
    return dnf_features()


def history_table(statuses: list[list[str]]) -> pd.DataFrame:
    """Past races with only the columns the counting needs, one list of statuses per race."""
    rows = [
        {"season": 2020, "round": rnd, "race_seq": rnd - 1, "driver_id": f"d{i}", "status_class": s}
        for rnd, race in enumerate(statuses, start=1)
        for i, s in enumerate(race)
    ]
    return pd.DataFrame(rows)


def backtest_auc(make_model, frame: pd.DataFrame, season: int = 2022) -> float:
    """Pooled ROC AUC of one model's backtest over a season of the made-up championship."""
    return season_auc(predict_races(make_model, frame, [season]), samples=100)["value"].iloc[0]


# ---------------------------------------------------------------------------
# Counting starts and retirements
# ---------------------------------------------------------------------------


def test_only_cars_that_started_are_counted() -> None:
    """DNS and DSQ are left out: race 1 has 3 starters (1 DNF), race 2 has 2 (2 DNFs)."""
    history = history_table([["finished", "dnf", "finished", "dns"], ["dnf", "dsq", "dnf", "dns"]])
    counts = race_counts(history)
    assert counts[["round", "starters", "dnfs"]].values.tolist() == [[1, 3, 1], [2, 2, 2]]


def test_the_field_rate_only_looks_at_earlier_races() -> None:
    """Before race 3: races 1-2 (3 DNFs in 5 starts); window 1 keeps only race 2 (2 in 2)."""
    counts = race_counts(
        history_table([["finished", "dnf", "finished"], ["dnf", "dnf"], ["dnf", "dnf"]])
    )
    assert rate_before(counts, 2020, 3, None) == pytest.approx(3 / 5)
    assert rate_before(counts, 2020, 3, 1) == pytest.approx(1.0)
    assert np.isnan(rate_before(counts, 2020, 1, None))
    assert rate_before(counts, 2021, 1, 2) == pytest.approx(4 / 4)


@pytest.mark.parametrize("window", [1, 3, 10, None])
def test_the_rolling_and_one_race_field_rates_agree(frame, window) -> None:
    """The vectorized version used for training rows matches the one used for prediction."""
    counts = race_counts(frame)
    one_by_one = [
        rate_before(counts, season, rnd, window)
        for season, rnd in counts[["season", "round"]].itertuples(index=False)
    ]
    np.testing.assert_allclose(rates_before(counts, window), one_by_one)


def test_training_rows_get_the_rate_from_before_their_race(frame) -> None:
    """Every row of race 10 gets the DNF share of races 0-9, never its own race's."""
    counts = race_counts(frame)
    rows = with_field_rate(frame, counts, None)
    earlier = frame[frame["race_seq"] < 10]
    expected = earlier["status_class"].eq("dnf").mean()
    assert rows.loc[frame["race_seq"] == 10, FIELD_RATE].unique() == pytest.approx([expected])
    assert rows.loc[frame["race_seq"] == 0, FIELD_RATE].isna().all()


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------


def test_the_base_rate_gives_every_car_the_recent_share(frame) -> None:
    """Fitted on 2020, race 20 (2021 round 1) gets the DNF share of the last 5 races of 2020."""
    model = BaseRateDNF(window=5).fit(frame[frame["season"] == 2020])
    predicted = model.predict(feature_rows(frame, 20))
    last_five = frame[frame["race_seq"].between(15, 19)]
    assert predicted["p_dnf"].unique() == pytest.approx(
        [last_five["status_class"].eq("dnf").mean()]
    )
    assert list(predicted.columns) == [*KEYS, "p_dnf"]


def test_a_model_trained_on_later_races_still_uses_only_earlier_ones(frame) -> None:
    """Trained on everything, the base rate for race 30 still ignores races 30 and later."""
    everything = BaseRateDNF(window=None).fit(frame)
    up_to = BaseRateDNF(window=None).fit(frame[frame["race_seq"] < 30])
    race = feature_rows(frame, 30)
    pd.testing.assert_frame_equal(everything.predict(race), up_to.predict(race))


def test_a_short_window_catches_a_jump_in_the_dnf_rate() -> None:
    """Rates double from 2022: ten races in, the last 5 races see it, all history barely does."""
    jumped = dnf_features(jump_at=40)
    history = jumped[jumped["race_seq"] < 50]
    race = feature_rows(jumped, 50)
    short = BaseRateDNF(window=5).fit(history).predict(race)["p_dnf"].iloc[0]
    every = BaseRateDNF(window=None).fit(history).predict(race)["p_dnf"].iloc[0]
    true_rate = jumped.loc[jumped["race_seq"].between(40, 49), "status_class"].eq("dnf").mean()
    assert short == pytest.approx(true_rate, abs=0.08)
    assert short > every + 0.1


@pytest.mark.parametrize("model_class", LEARNED)
def test_learned_models_find_the_planted_signal(frame, model_class) -> None:
    """Backtested over 2022, both learned models rank cars almost as well as the truth.

    Even the true chances only reach an AUC of about 0.66 here (most back-row cars still
    finish), so the bar is: clear of a coin toss and within 0.06 of that ceiling.
    """
    season = frame[frame["season"] == 2022]
    ceiling = roc_auc(season["true_p_dnf"].to_numpy(), season["status_class"].eq("dnf"))
    auc = backtest_auc(model_class, frame)
    assert auc > 0.6
    assert auc > ceiling - 0.06


def test_the_base_rate_cannot_tell_cars_apart(frame) -> None:
    """Every car in a race gets the same chance, so its AUC hovers around a coin toss."""
    predictions = predict_races(BaseRateDNF, frame, [2022])
    assert predictions.groupby("round")["p_dnf"].nunique().eq(1).all()
    assert backtest_auc(BaseRateDNF, frame) == pytest.approx(0.5, abs=0.15)


def test_the_logistic_coefficient_on_the_grid_is_the_largest(frame) -> None:
    """The planted signal is the grid; it should get the biggest (positive) coefficient."""
    model = LogisticDNF().fit(frame[frame["season"] <= 2021])
    top = model.coefficients().iloc[0]
    assert (top["feature"], top["coef"] > 0) == ("grid", True)


@pytest.mark.parametrize("model_class", LEARNED)
def test_non_starters_and_disqualified_cars_change_nothing(frame, model_class) -> None:
    """Adding DNS and DSQ rows to the training data leaves the fitted model unchanged."""
    history = frame[frame["season"] == 2020]
    noise = history.sample(40, random_state=1).assign(
        driver_id=lambda d: d["driver_id"] + "x", status_class=["dns", "dsq"] * 20
    )
    race = feature_rows(frame, 25)
    clean = model_class().fit(history).predict(race)
    with_noise = model_class().fit(pd.concat([history, noise])).predict(race)
    pd.testing.assert_frame_equal(clean, with_noise)


@pytest.mark.parametrize("kind", DNF_KINDS)
def test_training_twice_gives_identical_chances(frame, kind) -> None:
    """Every model is deterministic: same races in, same chances out, to the last bit."""
    history = frame[frame["season"] == 2020]
    race = feature_rows(frame, 22)
    first = make_dnf_model(kind).fit(history).predict(race)
    again = make_dnf_model(kind).fit(history).predict(race)
    assert np.array_equal(first["p_dnf"], again["p_dnf"])
    assert first["p_dnf"].between(0, 1).all()


def test_the_logistic_numbers_match_scikit_learn(frame) -> None:
    """Predicting from the kept numbers gives scikit-learn's own answer."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    history = frame[frame["season"] == 2020]
    model = LogisticDNF().fit(history)
    rows = with_field_rate(history, race_counts(history), model.window)
    rows = rows[rows["status_class"].isin(["finished", "dnf"])]
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ).fit(rows[DNF_FEATURES], rows["status_class"].eq("dnf"))
    race = feature_rows(frame, 21)
    rate = rate_before(model.counts, 2021, 2, model.window)
    expected = pipeline.predict_proba(race.assign(**{FIELD_RATE: rate})[DNF_FEATURES])[:, 1]
    np.testing.assert_allclose(model.predict(race)["p_dnf"], expected, rtol=1e-9)


def test_misuse_fails_loudly(frame) -> None:
    """Predicting before training, two races at once, an unknown kind, no starters at all."""
    race = feature_rows(frame, 30)
    for kind in DNF_KINDS:
        with pytest.raises(RuntimeError, match="fit"):
            make_dnf_model(kind).predict(race)
    with pytest.raises(ValueError, match="exactly one race"):
        BaseRateDNF().fit(frame).predict(pd.concat([race, feature_rows(frame, 31)]))
    with pytest.raises(ValueError, match="Unknown DNF model"):
        make_dnf_model("crystal_ball")
    with pytest.raises(ValueError, match="starters"):
        BaseRateDNF().fit(frame.assign(status_class="dns"))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_auc_matches_scikit_learn_including_ties() -> None:
    """Rank-based AUC equals roc_auc_score, ties counted as half; NaN with one class."""
    rng = np.random.default_rng(3)
    chances = rng.choice([0.1, 0.2, 0.3, 0.5], size=300)
    happened = rng.random(300) < chances
    assert roc_auc(chances, happened) == pytest.approx(roc_auc_score(happened, chances))
    assert np.isnan(roc_auc(chances, np.zeros(300, dtype=bool)))


def test_dnf_scores_leave_out_non_starters() -> None:
    """A DNS row given 100% would add a big error; it's ignored, like a DSQ row."""
    race = pd.DataFrame(
        {
            "status_class": ["finished", "dnf", "dns", "dsq"],
            "p_dnf": [0.2, 0.6, 1.0, 1.0],
        }
    )
    scores = dnf_scores(race)
    assert scores["brier_dnf"] == pytest.approx((0.2**2 + 0.4**2) / 2)
    assert scores["log_loss_dnf"] == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)
    assert np.isnan(dnf_scores(race.drop(columns="p_dnf"))["brier_dnf"])


def test_a_dnf_prediction_passes_the_backtest_checks(frame) -> None:
    """No finishing order is needed, but a chance above 100% is still refused."""
    race = frame[frame["race_seq"] == 30]
    good = race[KEYS].assign(p_dnf=0.1)
    check_prediction(good, race, "2021 round 11")
    with pytest.raises(PredictionError, match="p_dnf"):
        check_prediction(good.assign(p_dnf=1.5), race, "2021 round 11")


def test_dnf_chances_get_their_own_calibration_rows(frame) -> None:
    """Calibration bins for ``dnf`` cover exactly the cars that started."""
    predictions = predict_races(LogisticDNF, frame, [2022])
    table = calibration_table(predictions)
    assert set(table["outcome"]) == {"dnf"}
    assert table["n"].sum() == len(predictions)
    assert dnf_race_metrics(predictions)["starters"].sum() == len(predictions)


# ---------------------------------------------------------------------------
# Comparing and choosing on 2024
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison(frame):
    """The three models compared on 2021-2022 of the made-up championship."""
    return compare_dnf_models(frame, [2021, 2022], window=10)


def test_the_comparison_scores_every_model(comparison) -> None:
    """Brier, log loss and AUC for each model and season; differences only for learned models."""
    summary = comparison.summary
    assert set(summary["model"]) == set(DNF_KINDS)
    assert set(summary["metric"]) == {"brier_dnf", "log_loss_dnf", "auc_dnf"}
    assert len(summary) == 3 * 3 * 2
    assert set(comparison.differences["model"]) == {"logistic", "lightgbm"}
    learned = comparison.differences.query("metric == 'brier_dnf' and season == 2022")
    assert (learned["difference"] < 0).all()


def test_the_comparison_is_saved(comparison, tmp_path) -> None:
    """Predictions, per-race scores, the summary and the differences from the base rate."""
    folder = save_dnf_comparison(comparison, tmp_path / "dnf")
    assert {path.name for path in folder.iterdir()} == {
        "predictions.parquet",
        "races.csv",
        "summary.csv",
        "vs_base_rate.csv",
    }


def test_the_lowest_brier_on_the_validation_season_is_chosen(comparison) -> None:
    """With 2021 standing in for the validation season, a learned model beats the base rate."""
    assert tuning.choose_dnf_kind(comparison.summary, season=2021) in {"logistic", "lightgbm"}
    with pytest.raises(ValueError, match="No 2020"):
        tuning.choose_dnf_kind(comparison.summary, season=2020)


def test_the_window_choice_follows_a_jump_in_the_rate() -> None:
    """When the rate doubles at the start of 2022, a short window scores better there."""
    window, table = tuning.validation_dnf_window(
        dnf_features(jump_at=40), season=2022, windows=(3, None)
    )
    assert window == 3
    assert table["window"].tolist() == [3, None]
    assert table["brier_dnf"].iloc[0] < table["brier_dnf"].iloc[1]


def test_dnf_choices_are_never_made_on_a_test_season(frame) -> None:
    """Choosing the window or the model on 2025 or 2026 would leak their results."""
    for season in config.TEST_SEASONS:
        with pytest.raises(ValueError, match="test season"):
            tuning.validation_dnf_window(frame, season=season)
        with pytest.raises(ValueError, match="test season"):
            tuning.choose_dnf_kind(pd.DataFrame(), season=season)


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------

TRAINED_ON = {"races": 40, "starters": 800, "dnfs": 180, "first_race": "x", "last_race": "y"}


@pytest.mark.parametrize("kind", DNF_KINDS)
def test_a_loaded_dnf_model_predicts_exactly_like_the_saved_one(frame, tmp_path, kind) -> None:
    """Base rate, logistic regression and LightGBM all survive the round trip unchanged."""
    model = make_dnf_model(kind, window=10).fit(frame[frame["season"] <= 2021])
    save_dnf_model(model, TRAINED_ON, tmp_path)
    loaded, description = load_dnf_model(tmp_path)
    race = feature_rows(frame, 45)
    assert np.array_equal(model.predict(race)["p_dnf"], loaded.predict(race)["p_dnf"])
    assert (description["kind"], description["window"]) == (kind, 10)
    assert model_paths("dnf", tmp_path)[0].exists() == (kind == "lightgbm")


def test_switching_away_from_lightgbm_removes_its_tree_file(frame, tmp_path) -> None:
    """A dnf.txt left over from an earlier LightGBM model would no longer match dnf.json."""
    history = frame[frame["season"] == 2020]
    save_dnf_model(LightGBMDNF().fit(history), TRAINED_ON, tmp_path)
    save_dnf_model(BaseRateDNF().fit(history), TRAINED_ON, tmp_path)
    assert not model_paths("dnf", tmp_path)[0].exists()


def test_a_dnf_file_that_doesnt_match_is_refused(frame, tmp_path) -> None:
    """Coefficients that don't fit the feature list, or an unknown kind, won't load."""
    save_dnf_model(LogisticDNF().fit(frame[frame["season"] == 2020]), TRAINED_ON, tmp_path)
    meta_path = model_paths("dnf", tmp_path)[1]
    description = json.loads(meta_path.read_text(encoding="utf-8"))

    meta_path.write_text(json.dumps({**description, "features": ["grid"]}), encoding="utf-8")
    with pytest.raises(ModelFileError, match="coefficients"):
        load_dnf_model(tmp_path)

    meta_path.write_text(json.dumps({**description, "kind": "pickle"}), encoding="utf-8")
    with pytest.raises(ModelFileError, match="not a DNF model"):
        load_dnf_model(tmp_path)


def test_a_missing_dnf_model_says_how_to_make_one(tmp_path) -> None:
    """No files yet: the error names the command that trains the model."""
    with pytest.raises(FileNotFoundError, match="f1predict train"):
        load_dnf_model(tmp_path)
    with pytest.raises(RuntimeError, match="fit"):
        save_dnf_model(BaseRateDNF(), TRAINED_ON, tmp_path)


def test_every_model_kind_is_known_by_name() -> None:
    """The names used on the command line and in saved files map to the right classes."""
    assert [type(make_dnf_model(kind)) for kind in DNF_KINDS] == [
        BaseRateDNF,
        LogisticDNF,
        LightGBMDNF,
    ]
    assert partial(make_dnf_model, "logistic", 5)().window == 5
