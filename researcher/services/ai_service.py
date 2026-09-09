"""Retry/timeout/logging wrapper around the provided `ai.*` package.

This is the ONLY module in `researcher/` that imports `ai.*`. Every other
module reaches source-fetching and synthesis through `AIService`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai import sources as ai_sources
from ai import synthesizer as ai_synth
from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Source
from ai.sources import WebSearchProvider
from researcher.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_EXCEPTIONS = (ProviderError, httpx.HTTPError, httpx.TimeoutException)


def _retry_policy(settings: Settings) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """A tenacity retry decorator built from the given settings.

    Only retries transient provider/HTTP failures. `ValueError`s from bad
    input (e.g. an empty question) are not retryable and propagate immediately.
    """
    return retry(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential(
            multiplier=settings.retry_backoff_seconds,
            min=settings.retry_backoff_seconds,
            max=8,
        ),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
    )


class AIService:
    """Thin retrying/logging wrapper around `ai.*`. No business logic lives here."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def fetch_wikipedia(
        self, query: str, *, max_results: int, client: Any = None
    ) -> list[Source]:
        return await self._retrying_fetch(
            "wikipedia", ai_sources.fetch_wikipedia, query, max_results, client
        )

    async def fetch_arxiv(
        self, query: str, *, max_results: int, client: Any = None
    ) -> list[Source]:
        return await self._retrying_fetch(
            "arxiv", ai_sources.fetch_arxiv, query, max_results, client
        )

    async def fetch_web(
        self,
        query: str,
        *,
        max_results: int,
        client: Any = None,
        provider: WebSearchProvider | None = None,
    ) -> list[Source]:
        async def _fn(q: str, *, max_results: int, client: Any = None) -> list[Source]:
            return await ai_sources.fetch_web(
                q, max_results=max_results, provider=provider, client=client
            )

        return await self._retrying_fetch("web", _fn, query, max_results, client)

    async def _retrying_fetch(
        self,
        origin: str,
        fn: Callable[..., Any],
        query: str,
        max_results: int,
        client: Any,
    ) -> list[Source]:
        @_retry_policy(self._settings)
        async def _call() -> list[Source]:
            start = time.perf_counter()
            result = await fn(query, max_results=max_results, client=client)
            elapsed = time.perf_counter() - start
            logger.info(
                "fetch origin=%s query=%r results=%d elapsed=%.3fs",
                origin, query, len(result), elapsed,
            )
            logger.debug("fetch origin=%s titles=%s", origin, [s.title for s in result])
            return result

        return await _call()

    def synthesize(self, question: str, sources: list[Source]) -> AnswerWithCitations:
        """Synchronous — matches `ai.synthesizer.synthesize`'s real signature.

        Callers on the async path should invoke this via `asyncio.to_thread`.
        """

        @_retry_policy(self._settings)
        def _call() -> AnswerWithCitations:
            start = time.perf_counter()
            answer = ai_synth.synthesize(question, sources)
            elapsed = time.perf_counter() - start
            logger.info(
                "synthesize question=%r n_sources=%d elapsed=%.3fs",
                question, len(sources), elapsed,
            )
            return answer

        return _call()
