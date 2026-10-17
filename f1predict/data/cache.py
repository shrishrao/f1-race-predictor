"""Disk cache for API responses: one JSON file per request, written atomically.

The cache only stores and returns responses. Deciding whether a stored copy is fresh enough
is the caller's job (see ``JolpicaClient.get_json``), which keeps this module simple.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SAFE_TEXT = re.compile(r"^[A-Za-z0-9_-]+$")

# ---------------------------------------------------------------------------
# Cache entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheEntry:
    """A cached API response, where it came from, and when it was downloaded."""

    url: str
    fetched_at: datetime
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# JSON file cache
# ---------------------------------------------------------------------------


class JsonCache:
    """Store API responses as JSON files under one root folder.

    A request for ``2026/17/results.json`` with ``limit=100, offset=0`` is stored at
    ``<root>/2026/17/results__limit-100__offset-0.json``, so the files are easy to find
    and open by hand.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, path: str, params: Mapping[str, object] | None = None) -> Path:
        """Return the cache file for a request.

        Every path segment and query parameter must be plain letters, digits, ``-`` or
        ``_``. Anything else (``..``, slashes, drive letters) raises ``ValueError``, so a
        malformed request can never read or write outside the cache folder.
        """
        parts = path.strip("/").removesuffix(".json").split("/")
        for part in parts:
            if not _SAFE_TEXT.match(part):
                raise ValueError(f"Unsafe path segment in cache key: {part!r}")
        name = parts[-1]
        for key, value in sorted((params or {}).items()):
            key_text, value_text = str(key), str(value)
            if not (_SAFE_TEXT.match(key_text) and _SAFE_TEXT.match(value_text)):
                raise ValueError(f"Unsafe query parameter in cache key: {key_text}={value_text}")
            name += f"__{key_text}-{value_text}"
        return self.root.joinpath(*parts[:-1], f"{name}.json")

    def load(self, path: str, params: Mapping[str, object] | None = None) -> CacheEntry | None:
        """Return the cached response for a request, or None if it is missing or unreadable.

        An unreadable file (for example one left half-written by a crash in an older
        version) is treated as a cache miss and logged, so it simply gets downloaded again.
        """
        file = self.path_for(path, params)
        if not file.exists():
            return None
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
            return CacheEntry(
                url=record["url"],
                fetched_at=datetime.fromisoformat(record["fetched_at"]),
                payload=record["payload"],
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable cache file %s (%s)", file, exc)
            return None

    def save(
        self,
        path: str,
        params: Mapping[str, object] | None,
        *,
        url: str,
        payload: dict[str, Any],
        fetched_at: datetime,
    ) -> Path:
        """Write a response to the cache and return the file path.

        The JSON goes to a temporary file first and is then renamed over the target. A
        rename is atomic, so an interrupted run never leaves a half-written cache file.
        """
        file = self.path_for(path, params)
        file.parent.mkdir(parents=True, exist_ok=True)
        record = {"url": url, "fetched_at": fetched_at.isoformat(), "payload": payload}
        handle, tmp_name = tempfile.mkstemp(dir=file.parent, prefix=f".{file.stem}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as tmp_file:
                json.dump(record, tmp_file)
            os.replace(tmp_name, file)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return file
