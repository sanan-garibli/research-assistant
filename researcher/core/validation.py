"""Input validation at the researcher package's entry points."""

from __future__ import annotations

from researcher.core.errors import InvalidQuestionError

_ORIGIN_ALIASES = {"wiki": "wikipedia", "wikipedia": "wikipedia", "arxiv": "arxiv", "web": "web"}
ALL_ORIGINS = frozenset({"wikipedia", "arxiv", "web"})


def validate_question(question: str, *, max_length: int) -> str:
    """Reject empty or oversized questions; return the trimmed question."""
    q = question.strip()
    if not q:
        raise InvalidQuestionError("Question must not be empty.")
    if len(q) > max_length:
        raise InvalidQuestionError(f"Question too long ({len(q)} > {max_length} characters).")
    return q


def parse_sources(raw: str | None) -> set[str]:
    """Parse a `--sources wiki,arxiv` style string into a set of canonical origins.

    Returns all origins if `raw` is None/empty. Raises on unknown tokens.
    """
    if not raw:
        return set(ALL_ORIGINS)
    out: set[str] = set()
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in _ORIGIN_ALIASES:
            raise InvalidQuestionError(
                f"Unknown source {token!r}; expected one of wiki|wikipedia|arxiv|web."
            )
        out.add(_ORIGIN_ALIASES[token])
    if not out:
        raise InvalidQuestionError("--sources must name at least one source.")
    return out
