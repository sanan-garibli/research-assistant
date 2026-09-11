"""Tests for researcher.services.search_terms.wikipedia_search_candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researcher.services.search_terms import wikipedia_search_candidates

_QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "research_questions.json"


def test_candidates_drop_stopwords_and_try_longest_window_first():
    assert wikipedia_search_candidates("What is photosynthesis and what are its main stages?") == [
        "photosynthesis stages",
        "photosynthesis",
        "stages",
    ]


def test_candidates_keep_hyphenated_tokens_whole():
    candidates = wikipedia_search_candidates("How does CRISPR-Cas9 gene editing work at a molecular level?")

    assert candidates[0] == "CRISPR-Cas9 gene editing"
    assert "CRISPR-Cas9" in candidates
    assert "CRISPR" not in candidates


def test_candidates_slide_windows_left_to_right_within_a_size():
    candidates = wikipedia_search_candidates("What were the main causes of the 2008 financial crisis?")

    assert candidates[:2] == ["causes 2008 financial", "2008 financial crisis"]


def test_candidates_never_exceed_max_window_words():
    candidates = wikipedia_search_candidates(
        "How do transformer-based language models handle long context windows?", max_attempts=100
    )

    assert max(len(c.split()) for c in candidates) == 3


def test_candidates_are_deduplicated():
    candidates = wikipedia_search_candidates("fusion fusion fusion", max_attempts=100)

    assert len(candidates) == len(set(candidates))


def test_candidates_are_capped_at_max_attempts():
    question = "How do transformer-based language models handle long context windows?"

    assert len(wikipedia_search_candidates(question)) == 8
    assert len(wikipedia_search_candidates(question, max_attempts=3)) == 3


def test_stopword_only_question_falls_back_to_the_question_itself():
    assert wikipedia_search_candidates("What is it?") == ["What is it?"]


@pytest.mark.parametrize(
    "question_id, expected_term",
    [
        ("q1", "photosynthesis"),
        ("q2", "language models long"),
        ("q3", "2008 financial crisis"),
        ("q4", "fusion energy"),
        ("q5", "CRISPR-Cas9 gene editing"),
    ],
)
def test_sample_questions_produce_a_term_wikipedia_title_search_matches(question_id, expected_term):
    # Expected terms are the ones verified to return articles from Wikipedia's
    # prefix-matching opensearch endpoint on 2026-09-12.
    questions = {q["id"]: q["text"] for q in json.loads(_QUESTIONS_FILE.read_text())["questions"]}

    assert expected_term in wikipedia_search_candidates(questions[question_id])
