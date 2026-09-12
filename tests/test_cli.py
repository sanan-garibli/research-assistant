"""Tests for researcher.cli (argument parsing, formatting, dispatch)."""

from __future__ import annotations

import pytest

from ai.schemas import AnswerWithCitations, Citation, Source
import researcher.cli as cli_module
from researcher.core.errors import InvalidQuestionError
from researcher.models import ResearchSession, SourceOutcome


def _session(question="What is CRISPR?") -> ResearchSession:
    src = Source(title="CRISPR", url="https://example.com/crispr", snippet="s", origin="wikipedia")
    answer = AnswerWithCitations(
        question=question, answer="CRISPR edits genes [1].", citations=[Citation(index=1, source=src)]
    )
    outcomes = [SourceOutcome(origin="wikipedia", source_count=1, elapsed_seconds=0.1)]
    return ResearchSession(question=question, answer=answer, outcomes=outcomes)


def test_format_answer_includes_question_answer_and_references():
    text = cli_module.format_answer(_session())
    assert "Q: What is CRISPR?" in text
    assert "CRISPR edits genes [1]." in text
    assert "[1] (wikipedia) CRISPR" in text
    assert "https://example.com/crispr" in text


class FakeResearcher:
    def __init__(self, session=None, raises=None):
        self._session = session or _session()
        self._raises = raises
        self.calls: list[dict] = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        await self.aclose()

    async def aclose(self):
        self.closed = True

    async def ask(self, question, *, origins=None, use_cache=True):
        self.calls.append({"question": question, "origins": origins, "use_cache": use_cache})
        if self._raises:
            raise self._raises
        return self._session


class _FakeSettings:
    log_level = "INFO"


@pytest.fixture(autouse=True)
def patch_bootstrap(monkeypatch):
    monkeypatch.setattr(cli_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: _FakeSettings())


def test_ask_happy_path_prints_answer(monkeypatch, capsys):
    fake = FakeResearcher()
    monkeypatch.setattr(cli_module, "build_researcher", lambda settings: fake)

    exit_code = cli_module.main(["ask", "What is CRISPR?"])

    assert exit_code == 0
    assert "CRISPR edits genes" in capsys.readouterr().out
    assert fake.calls[0]["use_cache"] is True


def test_ask_no_cache_flag_propagates(monkeypatch):
    fake = FakeResearcher()
    monkeypatch.setattr(cli_module, "build_researcher", lambda settings: fake)

    cli_module.main(["ask", "q", "--no-cache"])

    assert fake.calls[0]["use_cache"] is False


def test_ask_sources_flag_restricts_origins(monkeypatch):
    fake = FakeResearcher()
    monkeypatch.setattr(cli_module, "build_researcher", lambda settings: fake)

    cli_module.main(["ask", "q", "--sources", "wiki,arxiv"])

    assert fake.calls[0]["origins"] == {"wikipedia", "arxiv"}


def test_ask_unknown_source_returns_error_exit_code(monkeypatch, capsys):
    fake = FakeResearcher()
    monkeypatch.setattr(cli_module, "build_researcher", lambda settings: fake)

    exit_code = cli_module.main(["ask", "q", "--sources", "reddit"])

    assert exit_code == 1
    assert "Error" in capsys.readouterr().err


def test_ask_empty_question_returns_error_exit_code(monkeypatch, capsys):
    fake = FakeResearcher(raises=InvalidQuestionError("Question must not be empty."))
    monkeypatch.setattr(cli_module, "build_researcher", lambda settings: fake)

    exit_code = cli_module.main(["ask", "   "])

    assert exit_code == 1
    assert "Error" in capsys.readouterr().err


def _session_with_subscript() -> ResearchSession:
    src = Source(title="Photosynthesis", url="https://example.com/p", snippet="s", origin="wikipedia")
    answer = AnswerWithCitations(
        question="What is photosynthesis?",
        answer="Plants convert CO₂ into sugars [1].",
        citations=[Citation(index=1, source=src)],
    )
    outcomes = [SourceOutcome(origin="wikipedia", source_count=1, elapsed_seconds=0.1)]
    return ResearchSession(question="What is photosynthesis?", answer=answer, outcomes=outcomes)


def test_ask_prints_non_latin1_characters_to_a_redirected_stream(monkeypatch):
    """Redirecting stdout on Windows gives a cp1252 stream, which cannot encode
    CO2's subscript; the CLI must not die with a UnicodeEncodeError."""
    import io

    fake = FakeResearcher(session=_session_with_subscript())
    monkeypatch.setattr(cli_module, "build_researcher", lambda settings: fake)
    raw = io.BytesIO()
    monkeypatch.setattr(cli_module.sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))

    exit_code = cli_module.main(["ask", "What is photosynthesis?"])

    cli_module.sys.stdout.flush()
    assert exit_code == 0
    assert "CO₂" in raw.getvalue().decode("utf-8")


def test_streams_that_cannot_be_reconfigured_are_left_alone(monkeypatch):
    import io

    fake = FakeResearcher()
    monkeypatch.setattr(cli_module, "build_researcher", lambda settings: fake)
    plain = io.StringIO()  # no reconfigure(), like pytest's captured stdout
    monkeypatch.setattr(cli_module.sys, "stdout", plain)

    exit_code = cli_module.main(["ask", "What is CRISPR?"])

    assert exit_code == 0
    assert "CRISPR edits genes" in plain.getvalue()


def test_demo_runs_limited_number_of_questions(monkeypatch, capsys):
    fake = FakeResearcher()
    monkeypatch.setattr(cli_module, "build_researcher", lambda settings: fake)

    exit_code = cli_module.main(["demo", "--limit", "2"])

    assert exit_code == 0
    assert len(fake.calls) == 2
