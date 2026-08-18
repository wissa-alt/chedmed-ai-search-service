"""Image-to-text catalogue search converging on the existing text pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from google import genai
from google.genai import types

from app.config import Settings
from models.search_query import SearchSource
from services.assistant_service import AssistantResponse

LOGGER = logging.getLogger(__name__)

IMAGE_DESCRIPTION_PROMPT = """You are a product image description system.

Describe only the visible product in one concise search-oriented phrase.
When visually identifiable, start with the concrete product type, then add visible
attributes such as color, shape, style, brand, or model only when clearly visible.
Never return attributes alone when the product type itself is visually identifiable.
Do not invent hidden attributes, price, city, condition, category, audience, brand,
or material that cannot be established from the image.
Do not recommend or select products. Return only the literal description."""

SELLER_IMAGE_ANALYSIS_PROMPT = """Analyze the seller's product image factually.
Identify the visible product type and describe only directly visible details such as
color, shape, visible logo or model text, connectors, and visible wear. Do not infer
material, condition, internal specifications, authenticity, accessories, price,
city, warranty, audience, or anything outside the image. Return one concise factual
paragraph and no sales advice."""


class ImageSearchError(RuntimeError):
    """Base error for image understanding and search."""


class InvalidImageError(ImageSearchError, ValueError):
    """Raised when an upload violates the image contract."""


class ImageProviderError(ImageSearchError):
    """Raised when the configured vision provider fails."""


class ImageDescriptionPort(Protocol):
    def describe(self, image: bytes, filename: str, content_type: str) -> str: ...
    def analyze(self, image: bytes, filename: str, content_type: str) -> str: ...


class TextAnswerPort(Protocol):
    def answer(
        self,
        query: str,
        top_k: int | None = None,
        *,
        source: SearchSource | str = SearchSource.TEXT,
    ) -> AssistantResponse: ...


@dataclass(frozen=True, slots=True)
class ImageSearchResult:
    description: str
    assistant_response: AssistantResponse
    provider: str
    vision_latency_ms: int
    total_latency_ms: int


class GeminiImageDescriptionProvider:
    """Describe original in-memory image bytes with the official Gemini SDK."""

    name = "gemini"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._model = settings.gemini_image_model
        self._api_key = settings.gemini_api_key
        self._client = client

    def describe(self, image: bytes, filename: str, content_type: str) -> str:
        return self._generate(image, filename, content_type, IMAGE_DESCRIPTION_PROMPT)

    def analyze(self, image: bytes, filename: str, content_type: str) -> str:
        return self._generate(image, filename, content_type, SELLER_IMAGE_ANALYSIS_PROMPT)

    def _generate(
        self, image: bytes, filename: str, content_type: str, prompt: str
    ) -> str:
        del filename
        if self._client is None:
            if not self._api_key:
                raise ImageProviderError("GEMINI_API_KEY est absente.")
            self._client = genai.Client(api_key=self._api_key)
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image, mime_type=content_type),
                ],
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=256),
            )
            description = response.text.strip()
        except Exception as exc:
            raise ImageProviderError("Gemini n'a pas pu décrire l'image.") from exc
        if not description:
            raise ImageProviderError("Gemini a retourné une description vide.")
        return description


class ImageSearchService:
    """Validate image, obtain a literal description, then reuse text search."""

    def __init__(
        self,
        provider: ImageDescriptionPort,
        text_service: TextAnswerPort,
        *,
        max_bytes: int,
        allowed_mime_types: frozenset[str],
    ) -> None:
        if max_bytes <= 0 or not allowed_mime_types:
            raise ValueError("La politique image est invalide.")
        self._provider = provider
        self._text_service = text_service
        self._max_bytes = max_bytes
        self._allowed_mime_types = allowed_mime_types

    def search(
        self,
        image: bytes,
        filename: str,
        content_type: str,
        top_k: int | None = None,
    ) -> ImageSearchResult:
        started = time.monotonic()
        self._validate(image, filename, content_type)
        vision_started = time.monotonic()
        description = self._provider.describe(image, filename, content_type)
        vision_latency = _elapsed_ms(vision_started)
        response = self._text_service.answer(
            description, top_k, source=SearchSource.IMAGE
        )
        total_latency = _elapsed_ms(started)
        LOGGER.info(
            "Image search completed: provider=%s bytes=%d mime=%s "
            "vision_latency_ms=%d total_latency_ms=%d results=%d",
            getattr(self._provider, "name", "unknown"),
            len(image),
            content_type,
            vision_latency,
            total_latency,
            len(response.search_result.items),
        )
        return ImageSearchResult(
            description,
            response,
            getattr(self._provider, "name", "unknown"),
            vision_latency,
            total_latency,
        )

    def describe(self, image: bytes, filename: str, content_type: str) -> str:
        """Expose validated visible observations for the seller assistant."""
        self._validate(image, filename, content_type)
        return self._provider.describe(image, filename, content_type)

    def analyze(self, image: bytes, filename: str, content_type: str) -> str:
        """Return detailed but visible-only observations for seller copy."""
        self._validate(image, filename, content_type)
        analyze = getattr(self._provider, "analyze", None)
        if callable(analyze):
            return analyze(image, filename, content_type)
        return self._provider.describe(image, filename, content_type)

    def _validate(self, image: bytes, filename: str, content_type: str) -> None:
        if not filename.strip():
            raise InvalidImageError("Le nom du fichier image est obligatoire.")
        if not image:
            raise InvalidImageError("Le fichier image est vide.")
        if len(image) > self._max_bytes:
            raise InvalidImageError("Le fichier image dépasse la taille maximale.")
        if content_type.lower() not in self._allowed_mime_types:
            raise InvalidImageError(f"Type MIME image non pris en charge: {content_type}.")


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
