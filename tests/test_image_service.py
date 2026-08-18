"""Tests for image understanding converging on the existing text pipeline."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.services.image_service import (
    GeminiImageDescriptionProvider,
    IMAGE_DESCRIPTION_PROMPT,
    ImageProviderError,
    ImageSearchService,
    InvalidImageError,
)
from config import Settings
from models.search_query import SearchSource
from search.search_service import SearchResult
from services.assistant_service import AssistantResponse


def _assistant_response(query: str = "black laptop") -> AssistantResponse:
    return AssistantResponse("Résultat catalogue.", SearchResult(query=query, items=()))


def _service(provider: Mock | None = None, text_service: Mock | None = None) -> ImageSearchService:
    provider_supplied = provider is not None
    provider = provider if provider_supplied else Mock(name="image_provider")
    provider.name = "gemini"
    if not provider_supplied:
        provider.describe.return_value = "black laptop"
    text_service_supplied = text_service is not None
    text_service = text_service if text_service_supplied else Mock(name="text_service")
    if not text_service_supplied:
        text_service.answer.return_value = _assistant_response()
    return ImageSearchService(
        provider,
        text_service,
        max_bytes=8,
        allowed_mime_types=frozenset({"image/png", "image/jpeg"}),
    )


def test_image_search_reuses_exact_text_pipeline() -> None:
    provider = Mock()
    provider.name = "gemini"
    provider.describe.return_value = "black laptop"
    text_service = Mock()
    response = _assistant_response()
    text_service.answer.return_value = response

    result = _service(provider, text_service).search(b"image", "item.png", "image/png", 4)

    provider.describe.assert_called_once_with(b"image", "item.png", "image/png")
    text_service.answer.assert_called_once_with(
        "black laptop", 4, source=SearchSource.IMAGE
    )
    assert result.description == "black laptop"
    assert result.assistant_response is response
    assert result.provider == "gemini"
    assert result.total_latency_ms >= result.vision_latency_ms >= 0


def test_image_description_preserves_visible_brand_for_text_search() -> None:
    provider = Mock(name="image_provider")
    provider.name = "gemini"
    provider.describe.return_value = "Dark gray Dell laptop"
    text_service = Mock(name="text_service")
    text_service.answer.return_value = _assistant_response("Dark gray Dell laptop")

    _service(provider, text_service).search(b"image", "laptop.png", "image/png", 5)

    text_service.answer.assert_called_once_with(
        "Dark gray Dell laptop", 5, source=SearchSource.IMAGE
    )


@pytest.mark.parametrize(
    ("image", "filename", "content_type", "message"),
    [
        (b"", "item.png", "image/png", "vide"),
        (b"image", "", "image/png", "nom"),
        (b"image", "item.gif", "image/gif", "MIME"),
        (b"123456789", "item.png", "image/png", "taille"),
    ],
)
def test_image_validation_rejects_invalid_uploads(
    image: bytes, filename: str, content_type: str, message: str
) -> None:
    with pytest.raises(InvalidImageError, match=message):
        _service().search(image, filename, content_type)


def test_image_service_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError, match="politique"):
        ImageSearchService(Mock(), Mock(), max_bytes=0, allowed_mime_types=frozenset())


def test_gemini_provider_sends_original_bytes_in_memory() -> None:
    response = Mock(text="black leather bag")
    client = Mock()
    client.models.generate_content.return_value = response
    provider = GeminiImageDescriptionProvider(
        Settings("test", "127.0.0.1", 8000, "INFO", gemini_image_model="gemini-test"),
        client=client,
    )

    assert provider.describe(b"original", "bag.png", "image/png") == "black leather bag"
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-test"
    assert call["contents"][1].inline_data.data == b"original"
    assert call["contents"][1].inline_data.mime_type == "image/png"


def test_image_prompt_requires_product_type_without_inventing_attributes() -> None:
    assert "start with the concrete product type" in IMAGE_DESCRIPTION_PROMPT
    assert "Never return attributes alone" in IMAGE_DESCRIPTION_PROMPT
    assert "Do not invent" in IMAGE_DESCRIPTION_PROMPT


def test_gemini_provider_rejects_empty_or_failed_response() -> None:
    client = Mock()
    client.models.generate_content.return_value = Mock(text="   ")
    provider = GeminiImageDescriptionProvider(
        Settings("test", "127.0.0.1", 8000, "INFO"), client=client
    )
    with pytest.raises(ImageProviderError, match="vide"):
        provider.describe(b"image", "item.png", "image/png")

    client.models.generate_content.side_effect = RuntimeError("provider down")
    with pytest.raises(ImageProviderError, match="décrire"):
        provider.describe(b"image", "item.png", "image/png")
