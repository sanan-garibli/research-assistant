"""Domain exception hierarchy for the researcher package."""

from __future__ import annotations


class ResearcherError(Exception):
    """Base class for all domain errors raised by the researcher package."""


class InvalidQuestionError(ResearcherError, ValueError):
    """Raised when a research question fails validation (empty or oversized)."""


class NoSourcesAvailableError(ResearcherError):
    """Raised when every requested source failed and no answer could be synthesized."""

    def __init__(self, question: str, failed_origins: list[str]) -> None:
        self.question = question
        self.failed_origins = failed_origins
        super().__init__(f"No sources available for {question!r}: {failed_origins}")
