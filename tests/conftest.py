"""Shared fixtures for Topic 4 smoke tests."""

from __future__ import annotations

from typing import Any

import pytest

from ai.providers.base import LLMProvider
from ai.sources import WebSearchProvider
from ai.schemas import Source


class FakeLLM(LLMProvider):
    """Returns a fixed text response. Records calls for inspection."""

    def __init__(self, response: str | None = None) -> None:
        self.response = response or (
            "Photosynthesis is the process by which plants convert light "
            "energy into chemical energy [1]. The reaction takes place in "
            "the chloroplasts and produces oxygen as a byproduct [2]."
        )
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(prompt)
        return self.response


class FakeWebSearch(WebSearchProvider):
    """Returns canned web results without touching the network."""

    def __init__(self, results: list[Source] | None = None) -> None:
        self.results = results or [
            Source(
                title="Photosynthesis — Encyclopedia",
                url="https://example.com/photosynthesis",
                snippet="A biological process used by plants and some bacteria.",
                origin="web",
            )
        ]
        self.calls: list[str] = []

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        client: Any = None,
    ) -> list[Source]:
        self.calls.append(query)
        return self.results[:max_results]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_web() -> FakeWebSearch:
    return FakeWebSearch()


@pytest.fixture
def sample_sources() -> list[Source]:
    return [
        Source(
            title="Photosynthesis (Wikipedia)",
            url="https://en.wikipedia.org/wiki/Photosynthesis",
            snippet="Photosynthesis is a process used by plants and other organisms "
                    "to convert light energy into chemical energy.",
            origin="wikipedia",
        ),
        Source(
            title="Calvin cycle (Wikipedia)",
            url="https://en.wikipedia.org/wiki/Calvin_cycle",
            snippet="The Calvin cycle is a series of biochemical redox reactions "
                    "in the stroma of chloroplasts.",
            origin="wikipedia",
        ),
    ]


# ---------------------------------------------------------------------------
# Fixtures for the student-built `researcher/` SE layer (additive; the
# fixtures above are the graded ai/ contract and must not be modified).
# ---------------------------------------------------------------------------

from researcher.config import Settings
from researcher.storage.cache_store import InMemoryCacheBackend


@pytest.fixture
def in_memory_cache_backend() -> InMemoryCacheBackend:
    return InMemoryCacheBackend()


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def settings_factory(cache_dir):
    def _make(**overrides) -> Settings:
        defaults = dict(
            cache_dir=cache_dir,
            cache_ttl_seconds=86400,
            per_source_timeout_seconds=5.0,
            max_sources_per_query=3,
            max_concurrent_fetches=5,
            max_question_length=500,
            retry_max_attempts=3,
            retry_backoff_seconds=0.01,
            tavily_api_key="test-key",
        )
        defaults.update(overrides)
        return Settings(**defaults)

    return _make


@pytest.fixture
def settings(settings_factory) -> Settings:
    return settings_factory()
