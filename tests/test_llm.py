"""Unit tests for Gemini LLM integration. No live API calls are made."""

from __future__ import annotations

import pytest

from src.generation.llm import LLMClient, LLMError, _friendly_gemini_error


def test_llm_client_raises_without_api_key():
    with pytest.raises(LLMError, match="Gemini API key"):
        LLMClient(api_key="", model="gemini-2.5-flash")


def test_friendly_gemini_error_maps_auth_errors():
    class FakeAuthenticationError(Exception):
        pass

    msg = _friendly_gemini_error(FakeAuthenticationError("401 unauthorized"))
    assert "API key" in msg


def test_friendly_gemini_error_maps_rate_limit_errors():
    class FakeResourceExhaustedError(Exception):
        pass

    msg = _friendly_gemini_error(FakeResourceExhaustedError("429 resource exhausted"))
    assert "rate limit" in msg.lower()


def test_friendly_gemini_error_maps_connection_errors():
    class FakeConnectionError(Exception):
        pass

    msg = _friendly_gemini_error(FakeConnectionError("network connection failed"))
    assert "connect" in msg.lower()


def test_friendly_gemini_error_falls_back_for_unknown_errors():
    msg = _friendly_gemini_error(ValueError("something obscure"))
    assert isinstance(msg, str)
    assert len(msg) > 0
