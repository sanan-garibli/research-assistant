"""Turn a natural-language question into search terms a title search can match.

The provided `ai.sources.fetch_wikipedia` uses Wikipedia's `opensearch`
endpoint, which matches article *titles by prefix*. A full sentence such as
"What is photosynthesis and what are its main stages?" never matches a title,
so every sample question came back with zero Wikipedia sources. `ai/` must not
be modified, so the SE layer reshapes the query instead: strip question words
and try short keyword phrases, most specific first.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# English function words plus words that are common in research questions but
# never start an article title ("main stages", "current state", "work at a
# molecular level", "handle long context").
_STOPWORDS = frozenset(
    """
    a about an and are as at be been by can could did do does for from had has have how i in
    into is it its may might of on or should that the their them there these they this those
    to was were what when where which who whom whose why will with would
    main current state level molecular work works handle handles
    """.split()
)


def wikipedia_search_candidates(
    question: str, *, max_window: int = 3, max_attempts: int = 8
) -> list[str]:
    """Ordered title-search terms for `question`, most specific first.

    Keywords are the question's tokens minus stopwords (hyphenated tokens such
    as "CRISPR-Cas9" stay whole). Candidates are contiguous keyword windows of
    at most `max_window` words, longest first and left to right within a size,
    deduplicated and capped at `max_attempts`. A question with no keywords
    falls back to the question itself.
    """
    keywords = [t for t in _TOKEN_RE.findall(question) if t.lower() not in _STOPWORDS]
    if not keywords:
        return [question]

    candidates: list[str] = []
    for size in range(min(max_window, len(keywords)), 0, -1):
        for start in range(len(keywords) - size + 1):
            term = " ".join(keywords[start : start + size])
            if term not in candidates:
                candidates.append(term)
    return candidates[:max_attempts]
