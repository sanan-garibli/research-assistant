"""Tests for researcher.config.Settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from researcher.config import Settings


def test_defaults_are_sensible():
    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"
    assert s.web_search_provider == "tavily"
    assert s.log_level == "INFO"
    assert s.max_concurrent_fetches == 5


def test_log_level_is_normalized_to_uppercase():
    s = Settings(_env_file=None, log_level="debug")
    assert s.log_level == "DEBUG"


def test_invalid_log_level_raises():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="not-a-level")


@pytest.mark.parametrize(
    "field", ["cache_ttl_seconds", "per_source_timeout_seconds", "max_sources_per_query",
              "max_concurrent_fetches", "max_question_length", "retry_max_attempts"]
)
def test_non_positive_numeric_fields_raise(field):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: 0})


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("MAX_SOURCES_PER_QUERY", "7")
    s = Settings(_env_file=None)
    assert s.llm_provider == "openai"
    assert s.max_sources_per_query == 7
