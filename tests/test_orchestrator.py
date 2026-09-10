"""Concurrency tests for researcher.concurrency.orchestrator.ResearchOrchestrator.

These exercise the mandatory concurrency behaviors: parallel fan-out (wall
clock ~= max, not sum), graceful degradation when one source raises, a
per-source timeout that doesn't block its siblings, and a bounded semaphore.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from researcher.concurrency.orchestrator import ResearchOrchestrator
from researcher.services.cache import SourceCache
from researcher.storage.cache_store import InMemoryCacheBackend
from ai.schemas import Source


def _source(origin: str) -> Source:
    return Source(title=f"{origin} title", url=f"https://example.com/{origin}", snippet="s", origin=origin)


class FakeAIService:
    """Stands in for AIService: one configurable async fetch per origin."""

    def __init__(self, behaviors: dict[str, dict]) -> None:
        # behaviors[origin] = {"delay": float, "raises": Exception|None, "result": list[Source]}
        self._behaviors = behaviors
        self.calls: list[str] = []
        self.active_windows: list[tuple[str, float, float]] = []

    async def _run(self, origin: str, query: str, *, max_results: int, client=None):
        self.calls.append(origin)
        b = self._behaviors.get(origin, {})
        start = time.perf_counter()
        delay = b.get("delay", 0.0)
        if delay:
            await asyncio.sleep(delay)
        self.active_windows.append((origin, start, time.perf_counter()))
        if b.get("raises"):
            raise b["raises"]
        return b.get("result", [_source(origin)])

    async def fetch_wikipedia(self, query, *, max_results=3, client=None):
        return await self._run("wikipedia", query, max_results=max_results, client=client)

    async def fetch_arxiv(self, query, *, max_results=3, client=None):
        return await self._run("arxiv", query, max_results=max_results, client=client)

    async def fetch_web(self, query, *, max_results=3, client=None):
        return await self._run("web", query, max_results=max_results, client=client)


def _orchestrator(ai_service, settings_factory, **settings_overrides):
    cache = SourceCache(InMemoryCacheBackend(), ttl_seconds=100)
    settings = settings_factory(**settings_overrides)
    return ResearchOrchestrator(ai_service, cache, settings), cache


@pytest.mark.asyncio
async def test_sources_fetched_in_parallel_not_sequentially(settings_factory):
    ai = FakeAIService({o: {"delay": 0.2} for o in ("wikipedia", "arxiv", "web")})
    orch, _ = _orchestrator(ai, settings_factory, per_source_timeout_seconds=5.0)

    start = time.perf_counter()
    outcomes, sources = await orch.gather_sources(
        "q", origins={"wikipedia", "arxiv", "web"}, use_cache=False
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.4  # ~= max(0.2s), not sum(0.6s)
    assert len(sources) == 3
    assert all(o.error is None for o in outcomes)


@pytest.mark.asyncio
async def test_one_source_raising_does_not_prevent_others(settings_factory):
    ai = FakeAIService(
        {
            "wikipedia": {"result": [_source("wikipedia")]},
            "arxiv": {"raises": RuntimeError("arxiv is down")},
            "web": {"result": [_source("web")]},
        }
    )
    orch, _ = _orchestrator(ai, settings_factory, per_source_timeout_seconds=5.0)

    outcomes, sources = await orch.gather_sources(
        "q", origins={"wikipedia", "arxiv", "web"}, use_cache=False
    )

    by_origin = {o.origin: o for o in outcomes}
    assert by_origin["arxiv"].error is not None
    assert by_origin["wikipedia"].error is None
    assert by_origin["web"].error is None
    assert {s.origin for s in sources} == {"wikipedia", "web"}


@pytest.mark.asyncio
async def test_slow_source_times_out_without_blocking_fast_sibling(settings_factory):
    ai = FakeAIService(
        {
            "wikipedia": {"delay": 5.0},  # will time out
            "arxiv": {"delay": 0.05, "result": [_source("arxiv")]},
        }
    )
    orch, _ = _orchestrator(ai, settings_factory, per_source_timeout_seconds=0.2)

    start = time.perf_counter()
    outcomes, sources = await orch.gather_sources(
        "q", origins={"wikipedia", "arxiv"}, use_cache=False
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0  # bounded by the timeout, not wikipedia's 5s delay
    by_origin = {o.origin: o for o in outcomes}
    assert by_origin["wikipedia"].error is not None
    assert by_origin["arxiv"].error is None
    assert {s.origin for s in sources} == {"arxiv"}


@pytest.mark.asyncio
async def test_semaphore_bounds_concurrent_fetches(settings_factory):
    ai = FakeAIService({o: {"delay": 0.1} for o in ("wikipedia", "arxiv", "web")})
    orch, _ = _orchestrator(
        ai, settings_factory, max_concurrent_fetches=1, per_source_timeout_seconds=5.0
    )

    await orch.gather_sources("q", origins={"wikipedia", "arxiv", "web"}, use_cache=False)

    windows = sorted(ai.active_windows, key=lambda w: w[1])
    for (_, _, end_a), (_, start_b, _) in zip(windows, windows[1:]):
        assert start_b >= end_a  # no two fetches overlapped


@pytest.mark.asyncio
async def test_cache_hit_skips_underlying_fetch(settings_factory):
    ai = FakeAIService({"wikipedia": {"result": [_source("wikipedia")]}})
    orch, cache = _orchestrator(ai, settings_factory, per_source_timeout_seconds=5.0)
    await cache.set("wikipedia", "q", [_source("wikipedia")])

    outcomes, sources = await orch.gather_sources("q", origins={"wikipedia"}, use_cache=True)

    assert ai.calls == []
    assert outcomes[0].from_cache is True
    assert len(sources) == 1


@pytest.mark.asyncio
async def test_http_client_is_built_once_and_reused_across_calls(settings_factory):
    """Building an httpx.AsyncClient costs ~0.4s (SSL trust store), so a
    per-question client would tax every query and discard connection pooling."""
    ai = FakeAIService({"wikipedia": {"result": [_source("wikipedia")]}})
    orch, _ = _orchestrator(ai, settings_factory, per_source_timeout_seconds=5.0)

    seen = []

    async def _capture(query, *, max_results=3, client=None):
        seen.append(client)
        return [_source("wikipedia")]

    orch._ai.fetch_wikipedia = _capture
    async with orch:
        for _ in range(3):
            await orch.gather_sources("q", origins={"wikipedia"}, use_cache=False)

    assert len(seen) == 3
    assert all(c is seen[0] for c in seen)
    assert seen[0] is not None


@pytest.mark.asyncio
async def test_aclose_releases_the_client_and_is_idempotent(settings_factory):
    ai = FakeAIService({"wikipedia": {"result": [_source("wikipedia")]}})
    orch, _ = _orchestrator(ai, settings_factory, per_source_timeout_seconds=5.0)

    await orch.gather_sources("q", origins={"wikipedia"}, use_cache=False)
    assert orch._client.is_closed is False

    await orch.aclose()
    await orch.aclose()  # no-op, must not raise

    assert orch._client.is_closed
