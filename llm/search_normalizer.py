"""Conservative Gemini normalization shared by every search source."""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types

from config import Settings

LOGGER = logging.getLogger(__name__)

SEARCH_NORMALIZATION_PROMPT = """You normalize marketplace search queries.

Return one short search query that preserves the user's actual product request.
Correct obvious spelling or speech-recognition errors and normalize Moroccan Darija,
Arabic, Arabizi, French, English, and mixed language when this clarifies the product
type. Preserve every brand, model, price, currency, color, audience, and city that is
actually present. Do not invent any attribute, product, intent, or catalogue fact.
Do not choose products and do not answer the user.

Original query:
{query}

Return only the normalized search query."""


class GeminiSearchNormalizer:
    """Normalize text with Gemini and fail open to the original query."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._model = settings.gemini_audio_model
        self._api_key = settings.gemini_api_key
        self._client = client

    def normalize(self, query: str) -> str:
        if self._client is None:
            if not self._api_key:
                return query
            self._client = genai.Client(api_key=self._api_key)
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=SEARCH_NORMALIZATION_PROMPT.format(query=query),
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=256),
            )
            normalized = response.text.strip()
        except Exception as exc:
            LOGGER.warning(
                "Gemini query normalization failed; original query preserved: %s",
                exc,
            )
            return query
        return " ".join(normalized.split()) or query
