"""Tests for project settings and folder overrides."""

from __future__ import annotations

from pathlib import Path

from f1predict import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_data_folders_are_absolute_and_inside_data_dir() -> None:
    """Every data folder is an absolute path inside DATA_DIR."""
    folders = (
        config.RAW_DIR,
        config.JOLPICA_CACHE_DIR,
        config.OPENF1_CACHE_DIR,
        config.INTERIM_DIR,
        config.PROCESSED_DIR,
        config.PUBLISHED_DIR,
        config.LOG_DIR,
    )
    for folder in folders:
        assert folder.is_absolute()
        assert config.DATA_DIR in folder.parents
    assert config.MODELS_DIR.is_absolute()


def test_env_var_moves_a_folder(monkeypatch, tmp_path) -> None:
    """Setting the environment variable points the folder somewhere else."""
    monkeypatch.setenv("F1PREDICT_TEST_DIR", str(tmp_path))
    assert config._dir_from_env("F1PREDICT_TEST_DIR", Path("unused")) == tmp_path.resolve()


def test_unset_or_blank_env_var_keeps_default(monkeypatch) -> None:
    """An unset or whitespace-only environment variable keeps the default folder."""
    default = Path("default-folder")
    monkeypatch.delenv("F1PREDICT_TEST_DIR", raising=False)
    assert config._dir_from_env("F1PREDICT_TEST_DIR", default) == default
    monkeypatch.setenv("F1PREDICT_TEST_DIR", "   ")
    assert config._dir_from_env("F1PREDICT_TEST_DIR", default) == default


# ---------------------------------------------------------------------------
# Seasons and defaults
# ---------------------------------------------------------------------------


def test_season_split_is_in_time_order() -> None:
    """Training data starts first, then the validation season, then the test seasons."""
    assert config.FIRST_SEASON < config.VALIDATION_SEASON < min(config.TEST_SEASONS)
    assert max(config.TEST_SEASONS) == config.LAST_SEASON


def test_simulation_defaults() -> None:
    """The default seed is 42 and the default number of simulated races is 100,000."""
    assert config.DEFAULT_SEED == 42
    assert config.DEFAULT_SIMS == 100_000
