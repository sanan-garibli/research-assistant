"""Business logic facade tying validation + orchestration + synthesis together."""

from __future__ import annotations

import asyncio

from researcher.config import Settings
from researcher.concurrency.orchestrator import ResearchOrchestrator
from researcher.core.errors import NoSourcesAvailableError
from researcher.core.validation import ALL_ORIGINS, validate_question
from researcher.models import ResearchSession
from researcher.services.ai_service import AIService


class Researcher:
    """Facade used by both the CLI and the benchmark script."""

    def __init__(
        self, orchestrator: ResearchOrchestrator, ai_service: AIService, settings: Settings
    ) -> None:
        self._orchestrator = orchestrator
        self._ai = ai_service
        self._settings = settings

    async def ask(
        self, question: str, *, origins: set[str] | None = None, use_cache: bool = True
    ) -> ResearchSession:
        question = validate_question(question, max_length=self._settings.max_question_length)
        origins = origins or set(ALL_ORIGINS)

        outcomes, sources = await self._orchestrator.gather_sources(
            question, origins=origins, use_cache=use_cache
        )

        if not sources:
            raise NoSourcesAvailableError(question, [o.origin for o in outcomes if o.error])

        answer = await asyncio.to_thread(self._ai.synthesize, question, sources)

        failed = [o for o in outcomes if o.error]
        if failed:
            note = "Note: " + "; ".join(f"{o.origin} unavailable ({o.error})" for o in failed)
            answer = answer.model_copy(update={"answer": f"{answer.answer}\n\n{note}"})

        return ResearchSession(question=question, answer=answer, outcomes=outcomes)
