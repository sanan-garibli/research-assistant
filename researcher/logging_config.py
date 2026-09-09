"""Structured logging setup. Call configure_logging() exactly once, at process startup."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib logging for the whole process.

    Diagnostics go through `logging.getLogger(__name__)` everywhere in this
    package; `print()` is reserved for CLI user-facing output only.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # Quiet the HTTP transport's own chatter; our own INFO logs already
    # cover per-source timing.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
