"""
LLM integration, isolated from the UI and from retrieval.

Reads OPENAI_API_KEY / OPENAI_MODEL from environment variables (via
utils.config) and never hard-codes credentials. The API key is never sent
to or rendered by the frontend -- it is only used server-side, inside this
process, when calling the OpenAI API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails for a reason the UI should surface."""


@dataclass
class GenerationResult:
    """The generated answer plus the chunks it was grounded in."""

    answer: str
    used_chunks: List[RetrievedChunk]


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError(
                "No OpenAI API key configured. Set OPENAI_API_KEY in your .env file."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "The `openai` package is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        history: List[dict],
        max_history_turns: int,
        temperature: float = 0.2,
    ) -> GenerationResult:
        """Generate a grounded answer to `question` using retrieved `chunks`.

        Args:
            question: The user's current question.
            chunks: Retrieved chunks to ground the answer in (may be empty).
            history: Prior conversation turns, for resolving follow-ups.
            max_history_turns: How many recent turns to include in the prompt.
            temperature: Sampling temperature (kept low for factual grounding).

        Returns:
            GenerationResult with the answer text and the chunks it was
            grounded in (i.e. the same `chunks` passed in, echoed back for
            convenience when rendering citations).

        Raises:
            LLMError: on authentication failure, rate limiting, or any other
                API error. Messages are written to be safe to show to users.
        """
        user_prompt = build_user_prompt(question, chunks, history, max_history_turns)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise LLMError(_friendly_openai_error(exc)) from exc

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise LLMError("The model returned an empty response. Please try again.")

        return GenerationResult(answer=answer, used_chunks=chunks)


def _friendly_openai_error(exc: Exception) -> str:
    """Translate common OpenAI SDK exceptions into user-facing messages."""
    # Import lazily/defensively: exact exception classes vary slightly by
    # SDK version, so we fall back to string matching if imports fail.
    name = type(exc).__name__
    message = str(exc)

    if "AuthenticationError" in name or "401" in message:
        return "The OpenAI API key appears to be invalid. Please check your .env file."
    if "RateLimitError" in name or "429" in message:
        return "The OpenAI API rate limit was reached. Please wait a moment and try again."
    if "APIConnectionError" in name or "Connection" in name:
        return "Could not connect to the OpenAI API. Please check your internet connection."
    if "APITimeoutError" in name or "Timeout" in name:
        return "The request to the OpenAI API timed out. Please try again."
    if "BadRequestError" in name or "400" in message:
        return "The request to the OpenAI API was invalid. This may be a configuration issue."

    logger.exception("Unhandled OpenAI API error")
    return "An unexpected error occurred while generating the answer. Please try again."
