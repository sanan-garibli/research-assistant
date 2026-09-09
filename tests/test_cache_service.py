"""Tests for researcher.services.cache (SourceCache, canonicalize_query, cache_key)."""

from __future__ import annotations

import time

import pytest

from researcher.services.cache import SourceCache, cache_key, canonicalize_query
from researcher.storage.cache_store import InMemoryCacheBackend


def test_canonicalize_query_normalizes_case_and_whitespace():
    assert canonicalize_query("  What Is   CRISPR? ") == "what is crispr?"


def test_cache_key_same_for_equivalent_queries():
    assert cache_key("wikipedia", "CRISPR") == cache_key("wikipedia", "  crispr ")


def test_cache_key_differs_by_source():
    assert cache_key("wikipedia", "q") != cache_key("arxiv", "q")


@pytest.mark.asyncio
async def test_get_returns_none_on_miss(in_memory_cache_backend, sample_sources):
    cache = SourceCache(in_memory_cache_backend, ttl_seconds=100)
    assert await cache.get("wikipedia", "q") is None


@pytest.mark.asyncio
async def test_set_then_get_round_trips(in_memory_cache_backend, sample_sources):
    cache = SourceCache(in_memory_cache_backend, ttl_seconds=100)
    await cache.set("wikipedia", "CRISPR", sample_sources)
    result = await cache.get("wikipedia", "  crispr ")
    assert result == sample_sources


@pytest.mark.asyncio
async def test_expired_entry_is_a_miss(in_memory_cache_backend, sample_sources):
    cache = SourceCache(in_memory_cache_backend, ttl_seconds=0.01)
    await cache.set("wikipedia", "q", sample_sources)
    time.sleep(0.05)
    assert await cache.get("wikipedia", "q") is None
