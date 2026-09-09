"""Typed configuration for the researcher package, loaded from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseSettings):
    """Typed settings read from `.env` / the environment.

    Note: `ai.providers.factory.get_llm()` and `ai.sources.get_web_search_provider()`
    read `os.environ` directly and know nothing about this object — `get_settings()`
    below calls `load_dotenv()` so both this Settings instance and those `ai.*`
    factories see the same values.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- LLM (for synthesis) ---
    llm_provider: str = "anthropic"
    llm_model: str | None = None
    anthropic_api_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)
    google_api_key: str | None = Field(default=None, repr=False)

    # --- Web search provider ---
    web_search_provider: str = "tavily"
    tavily_api_key: str | None = Field(default=None, repr=False)
    serper_api_key: str | None = Field(default=None, repr=False)

    # --- SE-layer settings ---
    log_level: str = "INFO"
    cache_dir: Path = Path("./.cache")
    cache_ttl_seconds: int = 86400
    per_source_timeout_seconds: float = 10.0
    max_sources_per_query: int = 3
    max_concurrent_fetches: int = 5
    max_question_length: int = 500
    retry_max_attempts: int = 3
    retry_backoff_seconds: float = 0.5

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in _VALID_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {v!r}")
        return upper

    @field_validator(
        "cache_ttl_seconds",
        "per_source_timeout_seconds",
        "max_sources_per_query",
        "max_concurrent_fetches",
        "max_question_length",
        "retry_max_attempts",
        "retry_backoff_seconds",
    )
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be a positive number")
        return v


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance (cached).

    Also loads `.env` into `os.environ` so the `ai.*` factories, which read
    environment variables directly, see the same configuration.
    """
    load_dotenv()
    return Settings()
