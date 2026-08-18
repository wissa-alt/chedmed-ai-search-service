"""Unit tests for the thin Flask adapters in the web composition root."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

from app.services.image_service import ImageSearchResult, InvalidImageError
from app.services.seller_assistant_service import (
    ComparableProduct,
    RecommendedRange,
    SellerAssistantResult,
    SellerAssistantUnavailableError,
)
from models.product import Product
from models.search_query import SearchSource
from models.transcription import TranscriptionResult
from runners.web import create_application
from search.search_service import SearchResult, SearchResultItem
from services.assistant_service import (
    AssistantResponse,
    AssistantServiceError,
    AudioAssistantResponse,
)
from sync.webhook_handler import InvalidWebhookError, SignatureError, WebhookResult


def _assistant_response() -> AssistantResponse:
    """Build one fully typed HTTP-search response fixture."""
    product = Product(
        id="product-1", title="Vélo", description="Description", category="Vélos",
        brand=None, color=None, condition=None, price=Decimal("100"), currency="MAD",
        city=None, image_urls=(), status="ACTIVE", is_sold=False,
        updated_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    result = SearchResult(query="vélo", items=(SearchResultItem(product, 0.9),))
    return AssistantResponse(answer="Voici un vélo.", search_result=result)


def test_health_requires_no_business_call() -> None:
    """The liveness endpoint remains independent of all providers."""
    application = create_application(Mock(), Mock())

    response = application.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_openapi_documents_all_public_search_endpoints() -> None:
    application = create_application(Mock(), Mock())

    response = application.test_client().get("/openapi.json")

    assert response.status_code == 200
    document = response.get_json()
    assert document["openapi"] == "3.1.0"
    assert set(("/health", "/search", "/transcriptions", "/voice-search", "/image-search", "/seller-assistant")) <= set(
        document["paths"]
    )
    assert document["paths"]["/search"]["post"]["requestBody"]["content"][
        "application/json"
    ]["example"] == {"query": "Dell laptop", "topK": 5}
    assert document["components"]["schemas"]["SearchResultItem"]["properties"][
        "product"
    ] == {"$ref": "#/components/schemas/Product"}


def test_openapi_exposes_binary_audio_and_image_uploads() -> None:
    application = create_application(Mock(), Mock())
    document = application.test_client().get("/openapi.json").get_json()
    voice_schema = document["paths"]["/voice-search"]["post"]["requestBody"][
        "content"
    ]["multipart/form-data"]["schema"]
    image_schema = document["paths"]["/image-search"]["post"]["requestBody"][
        "content"
    ]["multipart/form-data"]["schema"]

    assert voice_schema["properties"]["audio"] == {
        "type": "string",
        "format": "binary",
    }
    assert image_schema["properties"]["image"] == {
        "type": "string",
        "format": "binary",
    }
    assert "multipart/form-data" in document["paths"]["/voice-search"]["post"][
        "requestBody"
    ]["content"]
    assert "multipart/form-data" in document["paths"]["/image-search"]["post"][
        "requestBody"
    ]["content"]
    assert set(document["components"]["schemas"]) == {
        "ErrorResponse",
        "Product",
        "SearchResponse",
        "SearchResultItem",
        "UnderstoodQuery",
    }


def test_swagger_ui_and_alias_are_available() -> None:
    application = create_application(Mock(), Mock())
    client = application.test_client()

    docs = client.get("/docs")
    alias = client.get("/swagger")

    assert docs.status_code == 200
    assert b"SwaggerUIBundle" in docs.data
    assert b"/openapi.json" in docs.data
    assert alias.status_code == 302
    assert alias.headers["Location"].endswith("/docs")


def test_webhook_forwards_raw_body_and_serialises_result() -> None:
    """Flask is only an adapter; signature processing belongs to the handler."""
    handler = Mock()
    handler.handle.return_value = WebhookResult(True, False, "event-1", "synced", "Événement traité.")
    application = create_application(handler, Mock())

    response = application.test_client().post(
        "/webhooks/chedmed", data=b'{"eventId":"event-1"}', headers={"X-ChedMed-Signature": "abc"}
    )

    assert response.status_code == 200
    assert response.get_json()["eventId"] == "event-1"
    assert handler.handle.call_args.args[0] == b'{"eventId":"event-1"}'


def test_webhook_signature_error_is_http_401() -> None:
    """Only the adapter maps the domain signature exception to HTTP."""
    handler = Mock()
    handler.handle.side_effect = SignatureError("signature invalide")
    application = create_application(handler, Mock())

    assert application.test_client().post("/webhooks/chedmed", data=b"{}").status_code == 401


def test_search_serialises_assistant_result() -> None:
    """The route exposes answer text and typed product evidence."""
    assistant = Mock()
    assistant.answer.return_value = _assistant_response()
    application = create_application(Mock(), assistant)

    response = application.test_client().post("/search", json={"query": "vélo", "topK": 3})

    assert response.status_code == 200
    assert response.get_json()["results"][0]["product"]["id"] == "product-1"
    assistant.answer.assert_called_once_with(
        "vélo", 3, source=SearchSource.TEXT
    )


def test_search_rejects_non_object_or_service_error() -> None:
    """Invalid HTTP input and known business failures produce client responses."""
    assistant = Mock()
    assistant.answer.side_effect = AssistantServiceError("indisponible")
    application = create_application(Mock(), assistant)
    client = application.test_client()

    assert client.post("/search", json=[]).status_code == 400
    assert client.post("/search", json={"query": "vélo"}).status_code == 400


def test_transcription_reads_uploaded_audio_in_memory() -> None:
    """The route delegates uploaded bytes without a local temporary file."""
    assistant = Mock()
    assistant.transcribe.return_value = TranscriptionResult("bonjour", "whisper", False)
    application = create_application(Mock(), assistant)

    response = application.test_client().post(
        "/transcriptions", data={"audio": (io.BytesIO(b"audio"), "voice.wav")}
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "text": "bonjour",
        "provider": "whisper",
        "usedFallback": False,
        "resolutionReason": "single_provider",
        "qualityScore": 0.0,
        "latencyMs": 0,
    }
    assert assistant.transcribe.call_args.args[0] == b"audio"


def test_voice_search_reuses_search_response_format() -> None:
    """Multipart voice search exposes transcription plus standard search evidence."""
    assistant = Mock()
    assistant.answer_audio.return_value = AudioAssistantResponse(
        TranscriptionResult("gaming laptop", "whisper", False),
        _assistant_response(),
    )
    application = create_application(Mock(), assistant)
    response = application.test_client().post(
        "/voice-search",
        data={
            "audio": (io.BytesIO(b"audio"), "voice.wav"),
            "topK": "5",
        },
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["transcription"] == "gaming laptop"
    assert payload["provider"] == "whisper"
    assert payload["usedFallback"] is False
    assert payload["resolutionReason"] == "single_provider"
    assert payload["qualityScore"] == 0.0
    assert payload["transcriptionLatencyMs"] == 0
    assert payload["voiceSearchTotalLatencyMs"] == 0
    assert payload["alternativeTranscription"] is None
    assert payload["searchRecoveryUsed"] is False
    assert payload["searchEvidence"] == {}
    assert payload["whisperTranscription"] is None
    assert payload["geminiCorrection"] is None
    assert payload["whisperText"] is None
    assert payload["geminiText"] is None
    assert payload["geminiTriggered"] is False
    assert payload["geminiTriggerReason"] is None
    assert payload["whisperLatencyMs"] == 0
    assert payload["whisperSearchLatencyMs"] == 0
    assert payload["geminiLatencyMs"] == 0
    assert payload["geminiSearchLatencyMs"] == 0
    assert payload["answer"] == "Voici un vélo."
    assert payload["results"][0]["product"]["id"] == "product-1"
    assert assistant.answer_audio.call_args.args[3] == 5


def test_voice_search_rejects_invalid_top_k() -> None:
    application = create_application(Mock(), Mock())
    response = application.test_client().post(
        "/voice-search",
        data={"audio": (io.BytesIO(b"audio"), "voice.wav"), "topK": "zero"},
    )
    assert response.status_code == 400


def test_image_search_reuses_standard_search_response_format() -> None:
    """Image description is an input to the same assistant/search contract."""
    image_service = Mock()
    image_service.search.return_value = ImageSearchResult(
        "black laptop", _assistant_response(), "gemini", 12, 18
    )
    application = create_application(Mock(), Mock(), image_service)

    response = application.test_client().post(
        "/image-search",
        data={"image": (io.BytesIO(b"image"), "item.png"), "topK": "4"},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["query"] == "vélo"
    assert payload["answer"] == "Voici un vélo."
    assert payload["results"][0]["product"]["id"] == "product-1"
    assert payload["imageDescription"] == "black laptop"
    assert payload["provider"] == "gemini"
    assert payload["visionLatencyMs"] == 12
    assert payload["imageSearchTotalLatencyMs"] == 18
    image_service.search.assert_called_once_with(b"image", "item.png", "image/png", 4)


def test_image_search_accepts_file_alias_and_maps_validation_error() -> None:
    image_service = Mock()
    image_service.search.side_effect = InvalidImageError("type invalide")
    application = create_application(Mock(), Mock(), image_service)

    response = application.test_client().post(
        "/image-search", data={"file": (io.BytesIO(b"image"), "item.gif")}
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "type invalide"}


def test_image_search_is_explicitly_unavailable_without_service() -> None:
    application = create_application(Mock(), Mock())

    response = application.test_client().post("/image-search")

    assert response.status_code == 503


def test_seller_assistant_serialises_price_advice() -> None:
    seller_service = Mock()
    product = _assistant_response().search_result.items[0].product
    seller_service.assist.return_value = SellerAssistantResult(
        suggested_description="Vélo en bon état.",
        description_generated=True,
        description_quality="good",
        seller_price=Decimal("600"),
        currency="MAD",
        estimated_price=Decimal("420"),
        recommended_range=RecommendedRange(Decimal("400"), Decimal("450")),
        price_assessment="too_high",
        message="Votre prix semble nettement supérieur.",
        confidence="medium",
        comparables_count=3,
        comparables=(ComparableProduct(product, "relevant", 0.88),),
    )
    application = create_application(Mock(), Mock(), None, seller_service)

    response = application.test_client().post(
        "/seller-assistant",
        json={"title": "Vélo", "sellerPrice": 600, "currency": "mad"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["estimatedPrice"] == 420
    assert payload["recommendedRange"] == {"min": 400, "max": 450}
    assert payload["priceAssessment"] == "too_high"
    assert payload["comparables"][0]["id"] == "product-1"
    assert seller_service.assist.call_args.args[0].currency == "MAD"


def test_seller_assistant_accepts_optional_image_observations() -> None:
    image_service = Mock()
    image_service.describe.return_value = "gray Dell laptop"
    seller_service = Mock()
    seller_service.assist.return_value = SellerAssistantResult(
        suggested_description="Dell laptop gris.",
        description_generated=True,
        description_quality="good",
        seller_price=Decimal("600"), currency="MAD", estimated_price=None,
        recommended_range=None, price_assessment="insufficient_data",
        message="Données insuffisantes.", confidence="very_low",
        comparables_count=0, comparables=(),
    )
    application = create_application(Mock(), Mock(), image_service, seller_service)

    response = application.test_client().post(
        "/seller-assistant",
        data={
            "title": "Dell laptop",
            "sellerPrice": "600",
            "currency": "MAD",
            "image": (io.BytesIO(b"image"), "laptop.jpg"),
        },
    )

    assert response.status_code == 200
    image_service.describe.assert_called_once_with(b"image", "laptop.jpg", "image/jpeg")
    assert seller_service.assist.call_args.args[0].image_observations == "gray Dell laptop"


def test_seller_assistant_maps_validation_and_search_errors() -> None:
    seller_service = Mock()
    application = create_application(Mock(), Mock(), None, seller_service)
    client = application.test_client()

    assert client.post(
        "/seller-assistant",
        json={"title": "Laptop", "sellerPrice": 0, "currency": "MAD"},
    ).status_code == 400

    seller_service.assist.side_effect = SellerAssistantUnavailableError("indisponible")
    assert client.post(
        "/seller-assistant",
        json={"title": "Laptop", "sellerPrice": 100, "currency": "MAD"},
    ).status_code == 503


def test_seller_assistant_is_explicitly_unavailable_without_service() -> None:
    application = create_application(Mock(), Mock())
    assert application.test_client().post("/seller-assistant", json={}).status_code == 503


def test_simple_seller_description_supports_optional_image() -> None:
    seller = Mock()
    seller.suggest_description.return_value = ("Dell XPS gris proposé à la vente.", True)
    image = Mock()
    image.analyze.return_value = "Ordinateur portable gris avec logo Dell visible."
    app = create_application(Mock(), Mock(), image, seller)

    response = app.test_client().post(
        "/api/seller/suggest-description",
        data={
            "product_name": "Dell XPS", "category": "Électronique",
            "image": (io.BytesIO(b"image"), "laptop.jpg"),
        },
    )

    assert response.status_code == 200
    assert response.get_json()["image_analysis"].startswith("Ordinateur")
    image.analyze.assert_called_once_with(b"image", "laptop.jpg", "image/jpeg")


def test_simple_seller_price_endpoints_serialize_business_contracts() -> None:
    from app.services.seller_assistant_service import MarketStats, PriceCheckResult, PriceEstimateResult
    seller = Mock()
    product = _assistant_response().search_result.items[0].product
    seller.estimate_price.return_value = PriceEstimateResult(
        Decimal("400"), Decimal("450"), Decimal("350"), Decimal("500"),
        (ComparableProduct(product, "similar", 0.82),),
    )
    seller.check_price.return_value = PriceCheckResult(
        "too_low", "Prix très bas.", Decimal("2"),
        MarketStats(Decimal("450"), Decimal("400"), Decimal("350"), Decimal("500"), Decimal("300"), Decimal("700")),
        5,
    )
    client = create_application(Mock(), Mock(), None, seller).test_client()

    estimate = client.post("/api/seller/estimate-price", json={"description": "Laptop"})
    check = client.post("/api/seller/check-price", json={"description": "Laptop", "seller_price": 2})

    assert estimate.status_code == 200
    assert estimate.get_json()["suggested_price"] == 400
    assert estimate.get_json()["comparable_products"][0]["match_type"] == "similar"
    assert estimate.get_json()["comparable_products"][0]["status"] == product.status
    assert estimate.get_json()["comparable_products"][0]["isSold"] is False
    assert check.status_code == 200
    assert check.get_json()["market_stats"]["median"] == 400
    assert "comparable_products" in check.get_json()


def test_openapi_documents_three_simple_seller_endpoints() -> None:
    document = create_application(Mock(), Mock()).test_client().get("/openapi.json").get_json()
    assert {
        "/api/seller/suggest-description",
        "/api/seller/estimate-price",
        "/api/seller/check-price",
    } <= set(document["paths"])
    content = document["paths"]["/api/seller/suggest-description"]["post"]["requestBody"]["content"]
    assert "multipart/form-data" in content


def test_invalid_webhook_exception_is_http_400() -> None:
    """Invalid event payloads are represented as client errors by Flask only."""
    handler = Mock()
    handler.handle.side_effect = InvalidWebhookError("payload invalide")
    application = create_application(handler, Mock())

    assert application.test_client().post("/webhooks/chedmed", data=b"{}").status_code == 400
