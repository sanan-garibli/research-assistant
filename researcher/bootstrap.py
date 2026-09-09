"""Single composition root, shared by cli.py and scripts/bench.py."""

from __future__ import annotations

from researcher.concurrency.orchestrator import ResearchOrchestrator
from researcher.config import Settings, get_settings
from researcher.core.researcher import Researcher
from researcher.services.ai_service import AIService
from researcher.services.cache import SourceCache
from researcher.storage.cache_store import FilesystemCacheBackend


def build_researcher(settings: Settings | None = None) -> Researcher:
    settings = settings or get_settings()
    backend = FilesystemCacheBackend(settings.cache_dir)
    cache = SourceCache(backend, settings.cache_ttl_seconds)
    ai_service = AIService(settings)
    orchestrator = ResearchOrchestrator(ai_service, cache, settings)
    return Researcher(orchestrator, ai_service, settings)
