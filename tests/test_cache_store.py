"""Tests for researcher.storage.cache_store backends."""

from __future__ import annotations

import pytest

from researcher.storage.cache_store import FilesystemCacheBackend, InMemoryCacheBackend


@pytest.fixture(params=["filesystem", "memory"])
def backend(request, cache_dir):
    if request.param == "filesystem":
        return FilesystemCacheBackend(cache_dir)
    return InMemoryCacheBackend()


@pytest.mark.asyncio
async def test_write_then_read_round_trips(backend):
    payload = {"sources": [], "expires_at": 123.0}
    await backend.write("key1", payload)
    assert await backend.read("key1") == payload


@pytest.mark.asyncio
async def test_read_missing_key_returns_none(backend):
    assert await backend.read("does-not-exist") is None


@pytest.mark.asyncio
async def test_delete_removes_entry(backend):
    await backend.write("key1", {"a": 1})
    await backend.delete("key1")
    assert await backend.read("key1") is None


@pytest.mark.asyncio
async def test_delete_missing_key_does_not_raise(backend):
    await backend.delete("never-written")


@pytest.mark.asyncio
async def test_filesystem_backend_corrupted_file_is_treated_as_miss(cache_dir):
    backend = FilesystemCacheBackend(cache_dir)
    await backend.write("key1", {"a": 1})
    path = cache_dir / "sources" / "key1.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert await backend.read("key1") is None
