"""Tests for the JSON disk cache."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from f1predict.data.cache import JsonCache

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
PAYLOAD = {"MRData": {"total": "1", "RaceTable": {"Races": []}}}


def test_cache_file_name_is_readable_and_inside_the_root(tmp_path) -> None:
    """Path segments become folders and sorted query parameters become the file name."""
    path = JsonCache(tmp_path).path_for("2026/17/results.json", {"offset": 0, "limit": 100})
    assert path == tmp_path / "2026" / "17" / "results__limit-100__offset-0.json"


def test_save_then_load_round_trip(tmp_path) -> None:
    """A saved response comes back with its URL, payload and download time."""
    cache = JsonCache(tmp_path)
    cache.save(
        "2025.json", {"limit": 100}, url="https://example/2025.json", payload=PAYLOAD, fetched_at=T0
    )
    entry = cache.load("2025.json", {"limit": 100})
    assert entry is not None
    assert entry.payload == PAYLOAD
    assert entry.fetched_at == T0
    assert entry.url == "https://example/2025.json"


def test_missing_file_is_a_cache_miss(tmp_path) -> None:
    """Asking for something never saved returns None."""
    assert JsonCache(tmp_path).load("2025.json") is None


def test_unreadable_file_is_a_cache_miss_not_a_crash(tmp_path) -> None:
    """A corrupt cache file is ignored (and will simply be downloaded again)."""
    cache = JsonCache(tmp_path)
    file = cache.path_for("2025.json")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("{not valid json", encoding="utf-8")
    assert cache.load("2025.json") is None


@pytest.mark.parametrize(
    "bad_path", ["../secrets.json", "2025/../../x.json", "C:/Windows/x.json", "2025/res ults.json"]
)
def test_paths_that_could_escape_the_cache_are_rejected(tmp_path, bad_path: str) -> None:
    """Path traversal, drive letters and odd characters raise instead of touching other files."""
    with pytest.raises(ValueError, match="Unsafe"):
        JsonCache(tmp_path).path_for(bad_path)


def test_unsafe_query_parameters_are_rejected(tmp_path) -> None:
    """Query parameter values are checked too."""
    with pytest.raises(ValueError, match="Unsafe"):
        JsonCache(tmp_path).path_for("2025.json", {"limit": "../1"})


def test_saving_leaves_no_temporary_files(tmp_path) -> None:
    """The write-then-rename step cleans up after itself, even when overwriting."""
    cache = JsonCache(tmp_path)
    for _ in range(2):
        cache.save("2025.json", None, url="u", payload=PAYLOAD, fetched_at=T0)
    assert list(tmp_path.rglob("*.tmp")) == []
    assert [p.name for p in tmp_path.rglob("*.json")] == ["2025.json"]
