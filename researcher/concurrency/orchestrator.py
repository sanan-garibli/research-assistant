"""Concurrent multi-source orchestration: fan-out to wikipedia/arxiv/web with
per-source timeouts, a bounded semaphore, caching, and graceful degradation.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import ssl
import time

import httpx

from researcher.config import Settings
from researcher.models import SourceOutcome
from researcher.services.ai_service import AIService
from researcher.services.cache import SourceCache
from ai.schemas import Source

logger = logging.getLogger(__name__)

_ALL_ORIGINS = ("wikipedia", "arxiv", "web")
_ORIGIN_FETCHERS = {"wikipedia": "fetch_wikipedia", "arxiv": "fetch_arxiv", "web": "fetch_web"}


@functools.lru_cache(maxsize=1)
def _shared_ssl_context() -> ssl.SSLContext:
    """Build the trust store once per process.

    This is the ~0.35s that dominates `httpx.AsyncClient()` construction.
    An `SSLContext` is read-only once built and safe to share, so handing the
    same one to every client makes any client after the first ~free.
    """
    return httpx.create_ssl_context()


class ResearchOrchestrator:
    """Fans a question out to the requested sources concurrently.

    Owns a single, long-lived `asyncio.Semaphore` shared across every fetch
    issued through this instance, bounding total in-flight external calls
    (not just per-request concurrency).

    Also owns a single, long-lived `httpx.AsyncClient`, built here alongside
    the semaphore. A per-question client would throw away connection pooling
    between questions and re-pay client construction every time, so it is
    built once at composition time, off the query path, over the
    process-wide SSL context from `_shared_ssl_context()`. Construction
    needs no running event loop — the pool binds to one on first request —
    so a synchronous composition root can do this.

    The client must be released with `aclose()`; `async with orchestrator:`
    does that for you.
    """

    def __init__(self, ai_service: AIService, cache: SourceCache, settings: Settings) -> None:
        self._ai = ai_service
        self._cache = cache
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_fetches)
        self._client = httpx.AsyncClient(
            timeout=settings.per_source_timeout_seconds, verify=_shared_ssl_context()
        )

    async def __aenter__(self) -> "ResearchOrchestrator":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the shared HTTP client. Safe to call more than once."""
        await self._client.aclose()

    async def gather_sources(
        self, question: str, *, origins: set[str], use_cache: bool = True
    ) -> tuple[list[SourceOutcome], list[Source]]:
        requested = [o for o in _ALL_ORIGINS if o in origins]

        tasks = [self._run_one(origin, question, use_cache, self._client) for origin in requested]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outcomes: list[SourceOutcome] = []
        all_sources: list[Source] = []
        for origin, result in zip(requested, results):
            if isinstance(result, BaseException):
                outcomes.append(
                    SourceOutcome(
                        origin=origin, source_count=0, error=str(result), elapsed_seconds=0.0
                    )
                )
                continue
            outcome, sources = result
            outcomes.append(outcome)
            all_sources.extend(sources)
        return outcomes, all_sources

    async def _run_one(
        self, origin: str, question: str, use_cache: bool, client: httpx.AsyncClient
    ) -> tuple[SourceOutcome, list[Source]]:
        start = time.perf_counter()

        if use_cache:
            cached = await self._cache.get(origin, question)
            if cached is not None:
                elapsed = time.perf_counter() - start
                logger.info("cache hit origin=%s elapsed=%.4fs", origin, elapsed)
                return (
                    SourceOutcome(
                        origin=origin,
                        source_count=len(cached),
                        elapsed_seconds=elapsed,
                        from_cache=True,
                    ),
                    cached,
                )

        fetch = getattr(self._ai, _ORIGIN_FETCHERS[origin])
        try:
            async with self._semaphore:
                async with asyncio.timeout(self._settings.per_source_timeout_seconds):
                    sources = await fetch(
                        question, max_results=self._settings.max_sources_per_query, client=client
                    )
        except TimeoutError as e:
            elapsed = time.perf_counter() - start
            logger.warning("timeout origin=%s after=%.1fs", origin, elapsed)
            raise RuntimeError(
                f"{origin} timed out after {self._settings.per_source_timeout_seconds}s"
            ) from e

        elapsed = time.perf_counter() - start
        if use_cache:
            await self._cache.set(origin, question, sources)
        return SourceOutcome(origin=origin, source_count=len(sources), elapsed_seconds=elapsed), sources
