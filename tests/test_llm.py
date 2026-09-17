"""Tests for src.generation.llm.

These are unit tests only -- they do not call the real OpenAI API and do
not require an API key. Anything hitting the live API would be an
integration test (not included here, since it costs money and requires
credentials); see README.md for how to test the full pipeline manually.
"""

from __future__ import annotations

import pytest

from src.generation.llm import LLMClient, LLMError, _friendly_openai_error


def test_llm_client_raises_without_api_key():
    with pytest.raises(LLMError):
        LLMClient(api_key="", model="gpt-4o-mini")


def test_friendly_openai_error_maps_auth_errors():
    class FakeAuthenticationError(Exception):
        pass

    msg = _friendly_openai_error(FakeAuthenticationError("401 unauthorized"))
    assert "API key" in msg


def test_friendly_openai_error_maps_rate_limit_errors():
    class FakeRateLimitError(Exception):
        pass

    msg = _friendly_openai_error(FakeRateLimitError("429 too many requests"))
    assert "rate limit" in msg.lower()


def test_friendly_openai_error_falls_back_for_unknown_errors():
    msg = _friendly_openai_error(ValueError("something obscure"))
    assert isinstance(msg, str)
    assert len(msg) > 0
