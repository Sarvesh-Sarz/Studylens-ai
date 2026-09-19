"""Gemini LLM integration, isolated from the UI and retrieval."""

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
    """Thin wrapper around the Google GenAI generate-content API."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError(
                "No Gemini API key configured. Set GEMINI_API_KEY in your .env file."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise LLMError(
                "The `google-genai` package is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self.model = model
        try:
            self._client = genai.Client(api_key=api_key)
        except Exception as exc:
            raise LLMError(_friendly_gemini_error(exc)) from exc

    def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        history: List[dict],
        max_history_turns: int,
        temperature: float = 0.2,
    ) -> GenerationResult:
        """Generate a grounded answer using retrieved chunks and conversation history."""
        user_prompt = build_user_prompt(question, chunks, history, max_history_turns)

        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=temperature,
                ),
            )
        except Exception as exc:
            raise LLMError(_friendly_gemini_error(exc)) from exc

        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            raise LLMError("The model returned an empty response. Please try again.")

        return GenerationResult(answer=answer, used_chunks=chunks)


def _friendly_gemini_error(exc: Exception) -> str:
    """Translate common Gemini SDK/API errors into safe UI messages."""
    name = type(exc).__name__
    message = str(exc).lower()

    if "authentication" in name.lower() or "api key" in message or "401" in message:
        return "The Gemini API key appears to be invalid. Please check your .env file."
    if "permission" in name.lower() or "403" in message:
        return "The Gemini API request was not permitted. Check that the API key and project are configured correctly."
    if "resourceexhausted" in name.lower() or "ratelimit" in name.lower() or "429" in message:
        return "The Gemini API rate limit was reached. Please wait a moment and try again."
    if "timeout" in name.lower() or "timed out" in message:
        return "The request to the Gemini API timed out. Please try again."
    if "connection" in name.lower() or "network" in message:
        return "Could not connect to the Gemini API. Please check your internet connection."
    if "400" in message or "invalidargument" in name.lower():
        return "The request to the Gemini API was invalid. This may be a model or configuration issue."

    logger.exception("Unhandled Gemini API error")
    return "An unexpected error occurred while generating the answer. Please try again."
