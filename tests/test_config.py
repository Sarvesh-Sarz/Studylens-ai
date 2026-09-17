"""Tests for src.utils.config."""

from __future__ import annotations

import pytest

from src.utils.config import AppConfig, ConfigError


def test_default_config_has_sensible_values():
    config = AppConfig(openai_api_key="")
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 150
    assert config.top_k == 5
    assert config.min_relevance_score == 0.0
    assert config.openai_model == "gpt-5.6-terra"
    assert config.embedding_model == "all-MiniLM-L6-v2"


def test_validate_for_llm_raises_when_api_key_missing():
    config = AppConfig(openai_api_key="")
    with pytest.raises(ConfigError):
        config.validate_for_llm()


def test_validate_for_llm_passes_when_api_key_present():
    config = AppConfig(openai_api_key="sk-test-123")
    config.validate_for_llm()  # should not raise


def test_validate_chunking_rejects_zero_or_negative_chunk_size():
    config = AppConfig(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ConfigError):
        config.validate_chunking()


def test_validate_chunking_rejects_overlap_greater_than_or_equal_to_size():
    config = AppConfig(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ConfigError):
        config.validate_chunking()


def test_validate_chunking_passes_for_valid_values():
    config = AppConfig(chunk_size=1000, chunk_overlap=150)
    config.validate_chunking()  # should not raise


def test_validate_retrieval_rejects_non_positive_top_k():
    config = AppConfig(top_k=0)
    with pytest.raises(ConfigError):
        config.validate_retrieval()


def test_validate_retrieval_rejects_invalid_relevance_score():
    for score in (-1.01, 1.01):
        config = AppConfig(top_k=5, min_relevance_score=score)
        with pytest.raises(ConfigError):
            config.validate_retrieval()


def test_validate_retrieval_passes_for_valid_settings():
    config = AppConfig(top_k=5, min_relevance_score=0.75)
    config.validate_retrieval()  # should not raise
