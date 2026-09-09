"""Domain models for the researcher package. No naked dicts cross module boundaries."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ai.schemas import AnswerWithCitations


class SourceOutcome(BaseModel):
    """The result of trying to fetch one source (wikipedia/arxiv/web) for a question."""

    origin: str
    source_count: int
    error: str | None = None
    elapsed_seconds: float
    from_cache: bool = False


class ResearchSession(BaseModel):
    """A completed research question: the synthesized answer plus per-source outcomes."""

    question: str
    answer: AnswerWithCitations
    outcomes: list[SourceOutcome]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def degraded(self) -> bool:
        """True if at least one requested source failed to produce results."""
        return any(o.error for o in self.outcomes)
