"""Flask composition root and thin HTTP adapters for the ChedMed backend."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol

from flask import Flask, Response, jsonify, request

from app.config import Settings, get_settings
from app.services.catalog_service import create_catalogue_client
from app.services.embedding_service import EmbeddingService
from app.services.faiss_service import FAISSManager
from app.services.audio_service import (
    AudioProviderError,
    AudioTranscriptionError,
    AudioTranscriptionService,
    AudioValidationPolicy,
    GeminiAudioTranscriptionProvider,
    InvalidAudioError,
    TranscriptionQualityEvaluator,
    WhisperTranscriptionProvider,
)
from app.services.image_service import (
    GeminiImageDescriptionProvider,
    ImageProviderError,
    ImageSearchResult,
    ImageSearchService,
    InvalidImageError,
)
from app.services.seller_assistant_service import (
    PriceCheckResult,
    PriceEstimateResult,
    SellerAssistantResult,
    SellerAssistantService,
    SellerAssistantUnavailableError,
    SellerAssistantValidationError,
    SellerProductInput,
)
from app.services.text_service import (
    AssistantResponse,
    AssistantService,
    AssistantServiceError,
    QueryUnderstandingService,
    SearchService,
)
from llm.groq_client import GroqClient
from llm.search_normalizer import GeminiSearchNormalizer
from models.transcription import TranscriptionResult
from models.search_query import SearchSource
from services.assistant_service import AudioAssistantResponse
from services.sync_service import SynchronizationService
from sync.webhook_handler import InvalidWebhookError, SignatureError, WebhookHandler, WebhookResult
from search.query_expansion import QueryExpansionService
from search.product_text_builder import ProductTextBuilder
from runners.openapi import register_openapi
LOGGER = logging.getLogger(__name__)


class WebhookPort(Protocol):
    """Webhook dependency consumed by the HTTP adapter."""

    def handle(self, payload: bytes, headers: Mapping[str, str]) -> WebhookResult:
        """Process a signed webhook body."""


class AssistantPort(Protocol):
    """Assistant dependency consumed by search and transcription routes."""

    def answer(
        self, query: str, top_k: int | None = None, *,
        source: SearchSource | str = SearchSource.TEXT, include_all: bool = False,
    ) -> AssistantResponse:
        """Return a grounded catalogue response."""

    def transcribe(self, audio: bytes, filename: str, content_type: str) -> TranscriptionResult:
        """Return an audio transcription."""

    def answer_audio(
        self, audio: bytes, filename: str, content_type: str, top_k: int | None = None
    ) -> AudioAssistantResponse:
        """Return transcription plus the standard text-search response."""


class ImageSearchPort(Protocol):
    def search(
        self,
        image: bytes,
        filename: str,
        content_type: str,
        top_k: int | None = None,
    ) -> ImageSearchResult: ...

    def describe(self, image: bytes, filename: str, content_type: str) -> str: ...
    def analyze(self, image: bytes, filename: str, content_type: str) -> str: ...


class SellerAssistantPort(Protocol):
    def assist(self, seller: SellerProductInput) -> SellerAssistantResult: ...
    def suggest_description(
        self, product_name: str, category: str, keywords: str | None = None,
        language: str = "fr", image_analysis: str | None = None,
    ) -> tuple[str, bool]: ...
    def estimate_price(
        self, description: str, category: str | None = None, *, currency: str = "MAD"
    ) -> PriceEstimateResult: ...
    def check_price(
        self, description: str, seller_price: object, category: str | None = None,
        *, currency: str = "MAD",
    ) -> PriceCheckResult: ...


def create_application(
    webhook_handler: WebhookPort,
    assistant_service: AssistantPort,
    image_service: ImageSearchPort | None = None,
    seller_assistant_service: SellerAssistantPort | None = None,
) -> Flask:
    """Build a Flask app around fully injected business dependencies."""
    application = Flask(__name__)
    register_openapi(application)

    @application.get("/health")
    def health() -> tuple[Response, int]:
        """Return a liveness response without touching external dependencies."""
        return jsonify({"status": "ok"}), 200

    @application.post("/webhooks/chedmed")
    def chedmed_webhook() -> tuple[Response, int]:
        """Forward the untouched HTTP body and headers to the webhook handler."""
        try:
            result = webhook_handler.handle(request.get_data(cache=False), dict(request.headers))
        except SignatureError as exc:
            LOGGER.warning("Webhook ChedMed rejeté pour signature invalide: %s", exc)
            return jsonify({"error": str(exc)}), 401
        except InvalidWebhookError as exc:
            LOGGER.warning("Webhook ChedMed rejeté car invalide: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            LOGGER.exception("Erreur inattendue pendant le traitement du webhook ChedMed.")
            return jsonify({"error": "Erreur interne de traitement du webhook."}), 500
        status_code = 200 if result.accepted else 500
        return jsonify(_webhook_result_payload(result)), status_code

    @application.post("/search")
    def search() -> tuple[Response, int]:
        """Translate a JSON search request into an assistant-service call."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, Mapping):
            return jsonify({"error": "Le corps doit être un objet JSON."}), 400
        query = payload.get("query")
        top_k = payload.get("topK")
        include_all = payload.get("includeAll", False)
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool)):
            return jsonify({"error": "topK doit être un entier positif."}), 400
        if not isinstance(include_all, bool):
            return jsonify({"error": "includeAll doit être un booléen."}), 400
        try:
            if include_all:
                response = assistant_service.answer(
                    query, top_k, source=SearchSource.TEXT, include_all=True
                )
            else:
                response = assistant_service.answer(
                    query, top_k, source=SearchSource.TEXT
                )
        except (AssistantServiceError, ValueError) as exc:
            LOGGER.warning("Recherche HTTP invalide ou échouée: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            LOGGER.exception("Erreur inattendue pendant la recherche HTTP.")
            return jsonify({"error": "Erreur interne de recherche."}), 500
        return jsonify(_assistant_response_payload(response)), 200

    @application.post("/transcriptions")
    def transcriptions() -> tuple[Response, int]:
        """Pass an uploaded audio stream to the injected assistant service."""
        audio_file = request.files.get("audio")
        if audio_file is None:
            return jsonify({"error": "Le fichier audio est obligatoire."}), 400
        try:
            transcription = assistant_service.transcribe(
                audio_file.read(), audio_file.filename or "", audio_file.mimetype or ""
            )
        except InvalidAudioError as exc:
            LOGGER.warning("Transcription HTTP invalide ou échouée: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except (AudioProviderError, AudioTranscriptionError, AssistantServiceError) as exc:
            LOGGER.warning("Provider de transcription indisponible: %s", exc)
            return jsonify({"error": str(exc)}), 502
        except Exception:
            LOGGER.exception("Erreur inattendue pendant la transcription HTTP.")
            return jsonify({"error": "Erreur interne de transcription."}), 500
        return jsonify(_transcription_payload(transcription)), 200

    @application.post("/voice-search")
    def voice_search() -> tuple[Response, int]:
        """Transcribe uploaded audio, then invoke the existing search pipeline."""
        audio_file = request.files.get("audio")
        if audio_file is None:
            return jsonify({"error": "Le fichier audio est obligatoire."}), 400
        try:
            top_k = _multipart_top_k(request.form.get("topK"))
            response = assistant_service.answer_audio(
                audio_file.read(),
                audio_file.filename or "",
                audio_file.mimetype or "",
                top_k,
            )
        except (InvalidAudioError, ValueError) as exc:
            LOGGER.warning("Recherche vocale invalide: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except (AudioProviderError, AudioTranscriptionError, AssistantServiceError) as exc:
            LOGGER.warning("Recherche vocale échouée: %s", exc)
            return jsonify({"error": str(exc)}), 502
        except Exception:
            LOGGER.exception("Erreur inattendue pendant la recherche vocale.")
            return jsonify({"error": "Erreur interne de recherche vocale."}), 500
        payload = _assistant_response_payload(response.assistant_response)
        payload.update(
            {
                "transcription": response.transcription.text,
                "provider": response.transcription.primary_provider,
                "usedFallback": response.transcription.used_fallback,
                "resolutionReason": response.transcription.resolution_reason,
                "qualityScore": response.transcription.quality_score,
                "transcriptionLatencyMs": response.transcription.latency_ms,
                "voiceSearchTotalLatencyMs": response.latency_ms,
                "alternativeTranscription": response.transcription.alternative_text,
                "searchRecoveryUsed": response.transcription.search_recovery_used,
                "searchEvidence": dict(response.transcription.search_evidence),
                "whisperTranscription": response.transcription.whisper_text,
                "geminiCorrection": response.transcription.gemini_text,
                "whisperText": response.transcription.whisper_text,
                "geminiText": response.transcription.gemini_text,
                "geminiTriggered": response.transcription.gemini_triggered,
                "geminiTriggerReason": response.transcription.gemini_trigger_reason,
                "whisperLatencyMs": response.transcription.provider_latencies_ms.get(
                    "whisper", 0
                ),
                "whisperSearchLatencyMs": (
                    response.transcription.whisper_search_latency_ms
                ),
                "geminiLatencyMs": response.transcription.provider_latencies_ms.get(
                    "gemini", 0
                ),
                "geminiSearchLatencyMs": (
                    response.transcription.gemini_search_latency_ms
                ),
            }
        )
        return jsonify(payload), 200

    @application.post("/image-search")
    def image_search() -> tuple[Response, int]:
        """Describe an image, then reuse the exact grounded text-search flow."""
        if image_service is None:
            return jsonify({"error": "La recherche image est désactivée."}), 503
        image_file = request.files.get("image")
        if image_file is None:
            image_file = request.files.get("file")
        if image_file is None:
            return jsonify({"error": "Le fichier image est obligatoire."}), 400
        try:
            top_k = _multipart_top_k(request.form.get("topK"))
            result = image_service.search(
                image_file.read(),
                image_file.filename or "",
                image_file.mimetype or "",
                top_k,
            )
        except (InvalidImageError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except (ImageProviderError, AssistantServiceError) as exc:
            return jsonify({"error": str(exc)}), 502
        except Exception:
            LOGGER.exception("Erreur inattendue pendant la recherche image.")
            return jsonify({"error": "Erreur interne de recherche image."}), 500
        payload = _assistant_response_payload(result.assistant_response)
        payload.update(
            {
                "imageDescription": result.description,
                "provider": result.provider,
                "visionLatencyMs": result.vision_latency_ms,
                "imageSearchTotalLatencyMs": result.total_latency_ms,
            }
        )
        return jsonify(payload), 200

    @application.post("/seller-assistant")
    def seller_assistant() -> tuple[Response, int]:
        """Estimate a seller price from comparables returned by SearchService."""
        if seller_assistant_service is None:
            return jsonify({"error": "L'assistant vendeur est indisponible."}), 503
        payload: Mapping[str, Any] | None = request.get_json(silent=True)
        try:
            if request.mimetype == "multipart/form-data":
                multipart_payload = request.form.to_dict()
                image_file = request.files.get("image")
                if image_file is not None:
                    if image_service is None:
                        return jsonify({"error": "La compréhension image est désactivée."}), 503
                    multipart_payload["imageObservations"] = image_service.describe(
                        image_file.read(),
                        image_file.filename or "",
                        image_file.mimetype or "",
                    )
                payload = multipart_payload
            seller = SellerProductInput.from_mapping(payload)
            result = seller_assistant_service.assist(seller)
        except (SellerAssistantValidationError, InvalidImageError) as exc:
            return jsonify({"error": str(exc)}), 400
        except (SellerAssistantUnavailableError, ImageProviderError) as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception:
            LOGGER.exception("Erreur inattendue pendant l'assistance vendeur.")
            return jsonify({"error": "Erreur interne de l'assistant vendeur."}), 500
        return jsonify(_seller_assistant_payload(result)), 200

    @application.post("/api/seller/suggest-description")
    def seller_suggest_description() -> tuple[Response, int]:
        if seller_assistant_service is None:
            return jsonify({"error": "L'assistant vendeur est indisponible."}), 503
        try:
            product_name = request.form.get("product_name", "")
            category = request.form.get("category", "")
            keywords = request.form.get("keywords")
            language = request.form.get("language", "fr")
            image_analysis = None
            image_file = request.files.get("image")
            if image_file is not None and image_file.filename:
                if image_service is None:
                    return jsonify({"error": "La compréhension image est désactivée."}), 503
                image_analysis = image_service.analyze(
                    image_file.read(), image_file.filename, image_file.mimetype or ""
                )
            description, generated = seller_assistant_service.suggest_description(
                product_name, category, keywords, language, image_analysis
            )
        except (SellerAssistantValidationError, InvalidImageError) as exc:
            return jsonify({"error": str(exc)}), 400
        except ImageProviderError as exc:
            return jsonify({"error": str(exc)}), 502
        return jsonify({
            language or "fr": description,
            "image_analysis": image_analysis,
            "description_generated": generated,
        }), 200

    @application.post("/api/seller/estimate-price")
    def seller_estimate_price() -> tuple[Response, int]:
        if seller_assistant_service is None:
            return jsonify({"error": "L'assistant vendeur est indisponible."}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, Mapping):
            return jsonify({"error": "Le corps doit être un objet JSON."}), 400
        try:
            result = seller_assistant_service.estimate_price(
                payload.get("description"), payload.get("category"),
                currency=payload.get("currency", "MAD"),
            )
        except SellerAssistantValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except SellerAssistantUnavailableError as exc:
            return jsonify({"error": str(exc)}), 503
        return jsonify(_price_estimate_payload(result)), 200

    @application.post("/api/seller/check-price")
    def seller_check_price() -> tuple[Response, int]:
        if seller_assistant_service is None:
            return jsonify({"error": "L'assistant vendeur est indisponible."}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, Mapping):
            return jsonify({"error": "Le corps doit être un objet JSON."}), 400
        try:
            result = seller_assistant_service.check_price(
                payload.get("description"), payload.get("seller_price"),
                payload.get("category"), currency=payload.get("currency", "MAD"),
            )
        except SellerAssistantValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except SellerAssistantUnavailableError as exc:
            return jsonify({"error": str(exc)}), 503
        return jsonify(_price_check_payload(result)), 200

    return application


def build_application(settings: Settings) -> Flask:
    """Compose production infrastructure and expose it through Flask."""
    settings.ensure_runtime_directories()
    product_repository = create_catalogue_client(settings)
    text_builder = ProductTextBuilder()
    embedder = EmbeddingService(
        settings,
        text_builder=text_builder,
    )
    vector_store = FAISSManager(settings)

    synchronization_service = SynchronizationService(
        product_repository,
        embedder,
        vector_store,
        text_builder=text_builder,
    )
    webhook_handler = WebhookHandler(settings, synchronization_service)
    groq_client = GroqClient(settings)
    audio_mode = settings.audio_transcription_mode
    if audio_mode == "fallback" and not settings.audio_fallback_enabled:
        audio_mode = "whisper"
    audio_service = AudioTranscriptionService(
        WhisperTranscriptionProvider(groq_client),
        GeminiAudioTranscriptionProvider(settings),
        mode=audio_mode,
        validation_policy=AudioValidationPolicy(
            max_bytes=settings.audio_max_bytes,
            allowed_mime_types=frozenset(settings.audio_allowed_mime_types),
        ),
        evaluator=TranscriptionQualityEvaluator(
            quality_threshold=settings.audio_quality_threshold,
            max_no_speech_prob=settings.audio_max_no_speech_prob,
            min_avg_logprob=settings.audio_min_avg_logprob,
        ),
        log_transcripts=settings.audio_log_transcripts,
    )

    query_understanding = QueryUnderstandingService(
        groq_client,
        settings.groq_chat_model,
    )
    query_expansion = QueryExpansionService("resources/synonyms.json")
    search_service = SearchService(
        product_lookup=product_repository,
        embedder=embedder,
        vector_store=vector_store,
        query_understanding=query_understanding,
        query_expansion=query_expansion,
        default_top_k=settings.faiss_top_k_default,
        relevance_leader_margin=settings.relevance_leader_margin,
        relevance_max_relative_drop=settings.relevance_max_relative_drop,
        relevance_min_token_length=settings.relevance_min_token_length,
        query_normalizer=GeminiSearchNormalizer(settings),
    )
    assistant_service = AssistantService(
        search_service,
        groq_client,
        audio_service,
        search_recovery_enabled=settings.audio_fallback_enabled,
    )
    seller_assistant_service = SellerAssistantService(search_service, groq_client)
    image_service = None
    if settings.image_search_enabled:
        image_service = ImageSearchService(
            GeminiImageDescriptionProvider(settings),
            assistant_service,
            max_bytes=settings.image_max_bytes,
            allowed_mime_types=frozenset(settings.image_allowed_mime_types),
        )
    application = create_application(
        webhook_handler,
        assistant_service,
        image_service,
        seller_assistant_service,
    )
    application.extensions["catalogue_client"] = product_repository
    return application


def main() -> int:
    """Build and run the production HTTP server."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    application = build_application(settings)
    application.run(host=settings.host, port=settings.port)
    return 0


def production_application() -> Flask:
    """Build a fresh application for a WSGI server factory invocation."""
    return build_application(get_settings())


def _webhook_result_payload(result: WebhookResult) -> dict[str, Any]:
    """Serialise a webhook result without coupling it to Flask internals."""
    return {
        "accepted": result.accepted,
        "duplicated": result.duplicated,
        "eventId": result.event_id,
        "action": result.action,
        "message": result.message,
    }


def _assistant_response_payload(response: AssistantResponse) -> dict[str, Any]:
    """Serialise an assistant result and its evidence products."""
    return {
        "answer": response.answer,
        "query": response.search_result.query,
        "originalQuery": (
            response.search_result.original_query or response.search_result.query
        ),
        "normalizedQuery": (
            response.search_result.normalized_query or response.search_result.query
        ),
        "matchType": response.search_result.match_type,
        "primaryResultsCount": response.search_result.primary_results_count,
        "similarResultsCount": response.search_result.similar_results_count,
        "broadSimilarResultsCount": response.search_result.broad_similar_results_count,
        "totalCatalogProducts": response.search_result.total_catalog_products,
        "candidateProductsCount": response.search_result.candidate_products_count,
        "totalResultsCount": len(response.search_result.items),
        "understoodQuery": _structured_query_payload(response.search_result),
        "results": [
            {
                "product": item.product.to_dict(),
                "score": item.score,
                "semanticScore": item.score,
                "relevanceReason": item.relevance_reason,
                "lexicalTerms": list(item.lexical_terms),
                "matchType": item.match_type,
                "similarityScore": item.score,
            }
            for item in response.search_result.items
        ],
    }


def _structured_query_payload(result: Any) -> dict[str, Any] | None:
    query = result.structured_query
    if query is None:
        return None
    return {
        "category": query.category,
        "productType": query.product_type,
        "brand": query.brand,
        "color": query.color,
        "condition": query.condition,
        "city": query.city,
        "minPrice": query.min_price,
        "maxPrice": query.max_price,
        "currency": query.currency,
        "searchText": query.semantic_query,
    }


def _transcription_payload(result: TranscriptionResult) -> dict[str, Any]:
    """Expose only safe transcription provenance at the HTTP boundary."""
    return {
        "text": result.text,
        "provider": result.primary_provider,
        "usedFallback": result.used_fallback,
        "resolutionReason": result.resolution_reason,
        "qualityScore": result.quality_score,
        "latencyMs": result.latency_ms,
    }


def _multipart_top_k(raw_value: str | None) -> int | None:
    """Parse an optional positive multipart topK value."""
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("topK doit être un entier positif.") from exc
    if value <= 0:
        raise ValueError("topK doit être un entier positif.")
    return value


def _seller_assistant_payload(result: SellerAssistantResult) -> dict[str, Any]:
    price_range = result.recommended_range
    return {
        "suggestedDescription": result.suggested_description,
        "descriptionGenerated": result.description_generated,
        "descriptionQuality": result.description_quality,
        "sellerPrice": _json_number(result.seller_price),
        "currency": result.currency,
        "estimatedPrice": (
            _json_number(result.estimated_price)
            if result.estimated_price is not None else None
        ),
        "recommendedRange": (
            {
                "min": _json_number(price_range.minimum),
                "max": _json_number(price_range.maximum),
            }
            if price_range is not None else None
        ),
        "priceAssessment": result.price_assessment,
        "message": result.message,
        "confidence": result.confidence,
        "comparablesCount": result.comparables_count,
        "comparables": [
            {
                "id": comparable.product.id,
                "title": comparable.product.title,
                "price": _json_number(comparable.product.price),
                "currency": comparable.product.currency,
                "condition": comparable.product.condition,
                "matchType": comparable.match_type,
                "similarityScore": comparable.similarity_score,
            }
            for comparable in result.comparables
        ],
    }


def _price_estimate_payload(result: PriceEstimateResult) -> dict[str, Any]:
    return {
        "suggested_price": _json_number(result.suggested_price) if result.suggested_price is not None else None,
        "mean_price": _json_number(result.mean_price) if result.mean_price is not None else None,
        "price_range": (
            {"min": _json_number(result.minimum), "max": _json_number(result.maximum)}
            if result.minimum is not None and result.maximum is not None else None
        ),
        "comparable_products": [
            {
                "name": item.product.title,
                "price": _json_number(item.product.price),
                "similarity_score": item.similarity_score,
                "match_type": item.match_type,
                "status": item.product.status,
                "isSold": item.product.is_sold,
            }
            for item in result.comparables[:10]
        ],
        "based_on_n_products": len(result.comparables),
        "candidate_products_count": result.candidate_products_count,
        "total_catalog_products": result.total_catalog_products,
    }


def _price_check_payload(result: PriceCheckResult) -> dict[str, Any]:
    stats = result.stats
    return {
        "alert": result.alert,
        "message": result.message,
        "seller_price": _json_number(result.seller_price),
        "market_stats": (
            {
                "mean": _json_number(stats.mean), "median": _json_number(stats.median),
                "p25": _json_number(stats.p25), "p75": _json_number(stats.p75),
                "min": _json_number(stats.minimum), "max": _json_number(stats.maximum),
            }
            if stats is not None else None
        ),
        "based_on_n_products": result.comparables_count,
        "candidate_products_count": result.candidate_products_count,
        "total_catalog_products": result.total_catalog_products,
        "comparable_products": [
            {
                "name": item.product.title,
                "price": _json_number(item.product.price),
                "similarity_score": item.similarity_score,
                "match_type": item.match_type,
                "status": item.product.status,
                "isSold": item.product.is_sold,
            }
            for item in result.comparables[:10]
        ],
    }


def _json_number(value: Any) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


if __name__ == "__main__":
    raise SystemExit(main())
