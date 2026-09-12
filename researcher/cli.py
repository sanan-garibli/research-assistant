"""Command-line interface: `python -m researcher ask "..."` / `... demo`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from researcher.bootstrap import build_researcher
from researcher.config import get_settings
from researcher.core.errors import ResearcherError
from researcher.core.validation import parse_sources
from researcher.logging_config import configure_logging
from researcher.models import ResearchSession

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _use_utf8_stdio() -> None:
    """Make stdout/stderr able to carry any answer text.

    Answers are arbitrary text from upstream sources: a chemistry question
    comes back with "CO2" written using a subscript. When output is redirected
    on Windows, Python encodes it with the legacy code page (cp1252), which
    cannot represent that character, and `print` raises UnicodeEncodeError
    mid-answer. Streams that cannot be reconfigured (pytest's captured stdout,
    a plain StringIO) are left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):  # pragma: no cover - detached/odd stream
            pass


def format_answer(session: ResearchSession) -> str:
    """Pure rendering of a ResearchSession into human-readable text."""
    lines = [f"Q: {session.question}", "", f"A: {session.answer.answer}", ""]
    if session.answer.citations:
        lines.append("References:")
        for c in session.answer.citations:
            lines.append(f"  [{c.index}] ({c.source.origin}) {c.source.title}")
            lines.append(f"      {c.source.url}")
    timing = ", ".join(
        f"{o.origin}={o.elapsed_seconds:.2f}s{'*' if o.from_cache else ''}"
        for o in session.outcomes
    )
    lines += ["", f"(timing: {timing}; * = cache hit)"]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researcher", description="Async Research Assistant")
    parser.add_argument("--log-level", default=None, help="Override LOG_LEVEL")
    sub = parser.add_subparsers(dest="command", required=True)

    ask_p = sub.add_parser("ask", help="Ask a research question")
    ask_p.add_argument("question")
    ask_p.add_argument("--no-cache", action="store_true", help="Bypass the source cache")
    ask_p.add_argument(
        "--sources", default=None, help="Comma-separated subset: wiki,arxiv,web (default: all)"
    )
    ask_p.set_defaults(func=_cmd_ask)

    demo_p = sub.add_parser("demo", help="Run the sample research questions end-to-end")
    demo_p.add_argument("--limit", type=int, default=5)
    demo_p.add_argument("--no-cache", action="store_true")
    demo_p.set_defaults(func=_cmd_demo)

    return parser


async def _cmd_ask(args: argparse.Namespace, researcher) -> int:
    origins = parse_sources(args.sources)
    session = await researcher.ask(args.question, origins=origins, use_cache=not args.no_cache)
    print(format_answer(session))
    return 0


async def _cmd_demo(args: argparse.Namespace, researcher) -> int:
    questions = json.loads((_DATA_DIR / "research_questions.json").read_text())["questions"]
    for q in questions[: args.limit]:
        session = await researcher.ask(q["text"], use_cache=not args.no_cache)
        print(format_answer(session))
        print()
    return 0


async def _dispatch(args: argparse.Namespace, researcher) -> int:
    """Run the selected subcommand, releasing the researcher's resources after."""
    async with researcher:
        return await args.func(args, researcher)


def main(argv: list[str] | None = None) -> int:
    _use_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(args.log_level or settings.log_level)
    researcher = build_researcher(settings)

    try:
        return asyncio.run(_dispatch(args, researcher))
    except ResearcherError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
