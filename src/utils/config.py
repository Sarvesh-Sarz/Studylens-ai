"""
Centralized configuration for StudyLens AI.

All tunable parameters (chunking, retrieval, model names) are read from
environment variables so that nothing is hard-coded across the codebase.
Import `get_config()` wherever a setting is needed instead of reading
`os.environ` directly in other modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file if present (no-op in production if absent).
load_dotenv()


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}='{raw}' is not a valid integer.") from exc


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}='{raw}' is not a valid float.") from exc


@dataclass(frozen=True)
class AppConfig:
    """Immutable snapshot of application configuration."""

    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    # Chunking
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))

    # Retrieval
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 5))
    min_relevance_score: float = field(
        default_factory=lambda: _get_float("MIN_RELEVANCE_SCORE", 0.0)
    )

    # Conversation
    max_history_turns: int = field(default_factory=lambda: _get_int("MAX_HISTORY_TURNS", 6))

    # Storage
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))

    def validate_for_llm(self) -> None:
        """Raise ConfigError if settings required for LLM calls are missing."""
        if not self.openai_api_key:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment "
                "before asking questions."
            )

    def validate_chunking(self) -> None:
        """Raise ConfigError if chunking parameters are nonsensical."""
        if self.chunk_size <= 0:
            raise ConfigError("CHUNK_SIZE must be a positive integer.")
        if self.chunk_overlap < 0:
            raise ConfigError("CHUNK_OVERLAP cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    def validate_retrieval(self) -> None:
        """Raise ConfigError if retrieval parameters are nonsensical."""
        if self.top_k <= 0:
            raise ConfigError("TOP_K must be a positive integer.")


def get_config() -> AppConfig:
    """Return a fresh AppConfig built from the current environment.

    A fresh object is returned (rather than a cached singleton) so that
    tests can monkeypatch environment variables between calls.
    """
    return AppConfig()
