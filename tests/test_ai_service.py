"""Tests for researcher.services.ai_service.AIService (retries, logging)."""

from __future__ import annotations

import logging

import pytest

from ai.providers.base import ProviderError
from researcher.services import ai_service as ai_service_module
from researcher.services.ai_service import AIService


@pytest.mark.asyncio
async def test_fetch_wikipedia_retries_then_succeeds(monkeypatch, settings_factory, sample_sources):
    calls = {"n": 0}

    async def flaky(query, *, max_results=3, client=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ProviderError("transient failure")
        return sample_sources

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", flaky)
    service = AIService(settings_factory(retry_max_attempts=3, retry_backoff_seconds=0.001))

    result = await service.fetch_wikipedia("q", max_results=3)

    assert result == sample_sources
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_fetch_wikipedia_raises_after_exhausting_retries(monkeypatch, settings_factory):
    async def always_fails(query, *, max_results=3, client=None):
        raise ProviderError("permanent failure")

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", always_fails)
    service = AIService(settings_factory(retry_max_attempts=2, retry_backoff_seconds=0.001))

    with pytest.raises(ProviderError):
        await service.fetch_wikipedia("q", max_results=3)


@pytest.mark.asyncio
async def test_fetch_wikipedia_does_not_retry_on_value_error(monkeypatch, settings_factory):
    calls = {"n": 0}

    async def bad_input(query, *, max_results=3, client=None):
        calls["n"] += 1
        raise ValueError("bad input")

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", bad_input)
    service = AIService(settings_factory(retry_max_attempts=3, retry_backoff_seconds=0.001))

    with pytest.raises(ValueError):
        await service.fetch_wikipedia("q", max_results=3)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_fetch_logs_never_include_api_key(monkeypatch, settings_factory, sample_sources, caplog):
    async def fake(query, *, max_results=3, client=None):
        return sample_sources

    monkeypatch.setattr(ai_service_module.ai_sources, "fetch_wikipedia", fake)
    service = AIService(settings_factory(tavily_api_key="super-secret-key"))

    with caplog.at_level(logging.DEBUG):
        await service.fetch_wikipedia("q", max_results=3)

    assert "super-secret-key" not in caplog.text


def test_synthesize_wraps_ai_synthesizer(monkeypatch, settings_factory, sample_sources, fake_llm):
    from ai.synthesizer import synthesize as real_synthesize

    def synth(question, sources, *, llm=None):
        return real_synthesize(question, sources, llm=fake_llm)

    monkeypatch.setattr(ai_service_module.ai_synth, "synthesize", synth)
    service = AIService(settings_factory())

    answer = service.synthesize("What is photosynthesis?", sample_sources)

    assert answer.question == "What is photosynthesis?"
    assert answer.citations
