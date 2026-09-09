"""Sequential-vs-concurrent benchmark for the sample research questions.

Usage:
    python scripts/bench.py [--limit N]

Runs the same N questions twice, once sequentially (one `ask` after another)
and once concurrently (all N `ask` calls via asyncio.gather), both with
caching disabled so the numbers measure real fetch/synthesis concurrency
rather than cache behavior. Prints a Markdown table ready to paste into
README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from researcher.bootstrap import build_researcher
from researcher.config import get_settings
from researcher.logging_config import configure_logging

_QUESTIONS_FILE = _PROJECT_ROOT / "data" / "research_questions.json"


async def run_sequential(researcher, questions: list[str]) -> float:
    start = time.perf_counter()
    for q in questions:
        await researcher.ask(q, use_cache=False)
    return time.perf_counter() - start


async def run_concurrent(researcher, questions: list[str]) -> float:
    start = time.perf_counter()
    await asyncio.gather(*(researcher.ask(q, use_cache=False) for q in questions))
    return time.perf_counter() - start


def render_table(n: int, sequential: float, concurrent: float) -> str:
    speedup = sequential / concurrent if concurrent else float("inf")
    return (
        "| Workload | N | Sequential | Concurrent | Speedup |\n"
        "|---|---|---|---|---|\n"
        f"| 5 sample research questions | {n} | {sequential:.1f}s | {concurrent:.1f}s | {speedup:.1f}x |"
    )


async def _main(limit: int) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    questions = json.loads(_QUESTIONS_FILE.read_text())["questions"][:limit]
    texts = [q["text"] for q in questions]

    # Fresh Researcher instances per phase so no state (e.g. the semaphore or
    # the shared HTTP client) leaks between the sequential and concurrent
    # timing runs; each phase closes its own client when it finishes.
    async with build_researcher(settings) as researcher:
        sequential = await run_sequential(researcher, texts)
    async with build_researcher(settings) as researcher:
        concurrent = await run_concurrent(researcher, texts)

    print(render_table(len(texts), sequential, concurrent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential vs concurrent benchmark")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(_main(args.limit))


if __name__ == "__main__":
    main()
