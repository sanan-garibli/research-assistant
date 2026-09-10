"""Tests for researcher.core.researcher.Researcher (the facade / happy-path e2e)."""

from __future__ import annotations

import pytest
import pytest_asyncio

from ai.providers.base import ProviderError
from ai.schemas import Source
from ai.synthesizer import synthesize as real_synthesize
from researcher.concurrency.orchestrator import ResearchOrchestrator
from researcher.core.errors import InvalidQuestionError, NoSourcesAvailableError
from researcher.core.researcher import Researcher
from researcher.services import ai_service as ai_service_module
from researcher.services.ai_service import AIService
from researcher.services.cache import SourceCache
from researcher.storage.cache_store import InMemoryCacheBackend


def _source(origin: str) -> Source:
    return Source(title=f"{origin} title", url=f"https://example.com/{origin}", snippet="s", origin=origin)


@pytest_asyncio.fixture
async def wired_researcher(monkeypatch, settings_factory, fake_llm):
    """A Researcher wired to fakes for wikipedia/arxiv/web + a FakeLLM synthesizer."""
    settings = settings_factory(per_source_timeout_seconds=5.0)

    async def fake_wiki(query, *, max_results=3, client=None):
        return [_source("wikipedia")]

    async def fake_arxiv(query, *, max_results=3, client=None):
        return [_source("arxiv")]

    async def fake_web(query, *, max_results=3, provider=None, client=None):
        return [_source("web")]

    def fake_synth(question, sources, *, llm=None):
        return real_synthesize(question, sources, llm=fake_llm)

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", fake_wiki)
    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_arxiv", fake_arxiv)
    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_web", fake_web)
    monkeypatch.setattr(ai_service_module.ai_synth, "synthesize", fake_synth)

    ai_service = AIService(settings)
    cache = SourceCache(InMemoryCacheBackend(), ttl_seconds=settings.cache_ttl_seconds)
    orchestrator = ResearchOrchestrator(ai_service, cache, settings)
    async with Researcher(orchestrator, ai_service, settings) as researcher:
        yield researcher


@pytest.mark.asyncio
async def test_ask_happy_path_returns_session_with_citations(wired_researcher):
    session = await wired_researcher.ask("What is photosynthesis?", use_cache=False)

    assert session.question == "What is photosynthesis?"
    assert session.answer.citations
    assert {o.origin for o in session.outcomes} == {"wikipedia", "arxiv", "web"}
    assert session.degraded is False


@pytest.mark.asyncio
async def test_ask_rejects_empty_question(wired_researcher):
    with pytest.raises(InvalidQuestionError):
        await wired_researcher.ask("   ")


@pytest.mark.asyncio
async def test_ask_rejects_oversized_question(wired_researcher):
    with pytest.raises(InvalidQuestionError):
        await wired_researcher.ask("x" * 10_000)


@pytest.mark.asyncio
async def test_ask_restricts_to_requested_origins(wired_researcher):
    session = await wired_researcher.ask(
        "What is photosynthesis?", origins={"wikipedia"}, use_cache=False
    )
    assert {o.origin for o in session.outcomes} == {"wikipedia"}


@pytest.mark.asyncio
async def test_ask_raises_when_all_sources_fail(monkeypatch, settings_factory, fake_llm):
    settings = settings_factory(per_source_timeout_seconds=5.0)

    async def always_fails(query, *, max_results=3, client=None):
        raise ProviderError("down")

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", always_fails)
    ai_service = AIService(settings_factory(retry_max_attempts=1, per_source_timeout_seconds=5.0))
    cache = SourceCache(InMemoryCacheBackend(), ttl_seconds=settings.cache_ttl_seconds)
    orchestrator = ResearchOrchestrator(ai_service, cache, settings)

    async with Researcher(orchestrator, ai_service, settings) as researcher:
        with pytest.raises(NoSourcesAvailableError):
            await researcher.ask("q", origins={"wikipedia"}, use_cache=False)


@pytest.mark.asyncio
async def test_ask_partial_failure_notes_degradation_in_answer(
    monkeypatch, settings_factory, fake_llm
):
    settings = settings_factory(retry_max_attempts=1, per_source_timeout_seconds=5.0)

    async def fake_wiki(query, *, max_results=3, client=None):
        return [_source("wikipedia")]

    async def fails(query, *, max_results=3, client=None):
        raise ProviderError("arxiv down")

    def fake_synth(question, sources, *, llm=None):
        return real_synthesize(question, sources, llm=fake_llm)

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", fake_wiki)
    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_arxiv", fails)
    monkeypatch.setattr(ai_service_module.ai_synth, "synthesize", fake_synth)

    ai_service = AIService(settings)
    cache = SourceCache(InMemoryCacheBackend(), ttl_seconds=settings.cache_ttl_seconds)
    orchestrator = ResearchOrchestrator(ai_service, cache, settings)

    async with Researcher(orchestrator, ai_service, settings) as researcher:
        session = await researcher.ask("q", origins={"wikipedia", "arxiv"}, use_cache=False)

    assert session.degraded is True
    assert "arxiv" in session.answer.answer
