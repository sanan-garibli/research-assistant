"""Cache persistence backends.

`CacheBackend` mirrors the abstract-contract-plus-swappable-adapters pattern
already used by `ai.sources.WebSearchProvider`: a `SourceCache` (see
`researcher/services/cache.py`) is written against this interface only, so
swapping `FilesystemCacheBackend` for `InMemoryCacheBackend` in tests requires
no changes anywhere else.
"""

from __future__ import annotations

import abc
import asyncio
import json
from pathlib import Path
from typing import Any


class CacheBackend(abc.ABC):
    """Contract for cache persistence: read/write/delete a JSON-able payload by key."""

    @abc.abstractmethod
    async def read(self, key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abc.abstractmethod
    async def write(self, key: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def delete(self, key: str) -> None:
        raise NotImplementedError


class FilesystemCacheBackend(CacheBackend):
    """One JSON file per cache key under `cache_dir/sources/<key>.json`.

    File I/O runs in a thread via `asyncio.to_thread` so it never blocks the
    event loop that is concurrently fetching sources.
    """

    def __init__(self, cache_dir: Path | str) -> None:
        self._dir = Path(cache_dir) / "sources"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    async def read(self, key: str) -> dict[str, Any] | None:
        def _do() -> dict[str, Any] | None:
            path = self._path(key)
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A corrupted cache file is treated as a miss, not a crash.
                return None

        return await asyncio.to_thread(_do)

    async def write(self, key: str, payload: dict[str, Any]) -> None:
        def _do() -> None:
            self._path(key).write_text(json.dumps(payload), encoding="utf-8")

        await asyncio.to_thread(_do)

    async def delete(self, key: str) -> None:
        def _do() -> None:
            self._path(key).unlink(missing_ok=True)

        await asyncio.to_thread(_do)


class InMemoryCacheBackend(CacheBackend):
    """Dict-backed backend for fast, filesystem-free unit tests."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def read(self, key: str) -> dict[str, Any] | None:
        return self._store.get(key)

    async def write(self, key: str, payload: dict[str, Any]) -> None:
        self._store[key] = payload

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
