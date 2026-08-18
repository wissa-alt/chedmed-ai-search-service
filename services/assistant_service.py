
"""Application service for grounded catalogue answers and voice transcription."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, replace
from typing import Protocol

from models.product import Product
from models.search_query import SearchSource
from models.transcription import TranscriptionResult
from search.search_service import SearchResult, SearchService
from search.product_filter import _normalise_text
from services.audio_errors import AudioTranscriptionError

LOGGER = logging.getLogger(__name__)


class AssistantServiceError(RuntimeError):
    """Raised when an assistant workflow cannot produce a safe response."""


class CatalogueAnswerPort(Protocol):
    """Generate a catalogue-grounded answer from products."""

    def generate_catalogue_answer(
        self,
        query: str,
        products: tuple[Product, ...],
    ) -> str:
        """Return a textual answer for the query and products."""


class TranscriptionPort(Protocol):
    """Transcribe an uploaded audio recording."""

    def transcribe(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
    ) -> TranscriptionResult:
        """Return recognised speech and provider provenance."""

    def transcribe_alternative(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        *,
        provider: str = "gemini",
    ) -> TranscriptionResult:
        """Return an independent transcript from the same original audio."""

    def transcribe_whisper_primary(
        self, audio: bytes, filename: str, content_type: str
    ) -> TranscriptionResult:
        """Return the first-pass Whisper hypothesis and technical quality."""

    def review_with_gemini(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        whisper_text: str | None,
    ) -> TranscriptionResult:
        """Review Whisper against the same original audio."""


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """Grounded answer plus the search result that supports it."""

    answer: str
    search_result: SearchResult


@dataclass(frozen=True, slots=True)
class AudioAssistantResponse:
    """Voice transcription and the existing grounded text-search response."""

    transcription: TranscriptionResult
    assistant_response: AssistantResponse
    latency_ms: int = 0


class AssistantService:
    """Orchestrate search, answer generation, and optional voice transcription."""

    def __init__(
        self,
        search_service: SearchService,
        answer_client: CatalogueAnswerPort,
        transcription_client: TranscriptionPort,
        *,
        search_recovery_enabled: bool = True,
    ) -> None:
        """Store all injected collaborators."""
        self._search_service = search_service
        self._answer_client = answer_client
        self._transcription_client = transcription_client
        self._search_recovery_enabled = search_recovery_enabled

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        *,
        source: SearchSource | str = SearchSource.TEXT,
        include_all: bool = False,
    ) -> AssistantResponse:
        """Return a grounded answer for the query.

        All collaborator failures are translated into
        ``AssistantServiceError`` so infrastructure or provider
        exceptions do not leak through the application boundary.
        """
        try:
            LOGGER.info("Début du traitement de la requête assistant.")

            if include_all:
                search_result = self._search_service.search(
                    query, top_k, source=source, include_all=True
                )
            else:
                search_result = self._search_service.search(query, top_k, source=source)

            LOGGER.info(
                "Recherche terminée : %d produit(s) trouvé(s).",
                len(search_result.items),
            )

            products = tuple(
                item.product
                for item in search_result.items
            )

            if include_all:
                # Full-catalogue mode is an inspection/ranking endpoint. Do
                # not send an unbounded catalogue (including unrelated
                # products) to the answer LLM: the ranked result list is the
                # source of truth and must remain available even if the answer
                # provider has a smaller context window.
                answer = (
                    f"Catalogue classé pour cette recherche : "
                    f"{len(products)} produit(s)."
                )
            elif products:
                answer = self._answer_client.generate_catalogue_answer(
                    search_result.query,
                    products,
                )
                if search_result.match_type == "similar":
                    answer = (
                        "Aucune correspondance suffisamment précise n'a été trouvée. "
                        "Voici des produits similaires.\n\n"
                        f"{answer}"
                    )
                elif search_result.match_type == "broad_similar":
                    answer = (
                        "Aucune correspondance précise ou de même famille n'a été trouvée. "
                        "Voici des produits plus largement similaires.\n\n"
                        f"{answer}"
                    )
                elif (
                    search_result.similar_results_count
                    or search_result.broad_similar_results_count
                ):
                    answer = (
                        "Voici les produits correspondant à votre recherche, "
                        "suivis de produits similaires.\n\n"
                        f"{answer}"
                    )
            else:
                answer = "Aucun produit suffisamment pertinent ou similaire n'a été trouvé."

            LOGGER.info("Réponse catalogue générée avec succès.")

            return AssistantResponse(
                answer=answer,
                search_result=search_result,
            )

        except Exception as exc:
            LOGGER.exception(
                "Le traitement de la réponse assistant a échoué."
            )
            raise AssistantServiceError(
                "Impossible de générer une réponse assistant."
            ) from exc

    def transcribe(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
    ) -> TranscriptionResult:
        """Return voice transcription via the injected transcription client."""
        try:
            return self._transcription_client.transcribe(
                audio,
                filename,
                content_type,
            )
        except AudioTranscriptionError:
            raise
        except Exception as exc:
            LOGGER.exception(
                "La transcription du service assistant a échoué."
            )
            raise AssistantServiceError(
                "Impossible de transcrire le fichier audio."
            ) from exc

    def answer_audio(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        top_k: int | None = None,
    ) -> AudioAssistantResponse:
        """Run Whisper search, then one audio-grounded Gemini review if needed."""
        started = time.monotonic()
        whisper: TranscriptionResult | None = None
        whisper_response: AssistantResponse | None = None
        whisper_search_latency = 0
        try:
            whisper = self._transcription_client.transcribe_whisper_primary(
                audio, filename, content_type
            )
            search_started = time.monotonic()
            whisper_response = self.answer(
                whisper.text, top_k, source=SearchSource.AUDIO
            )
            whisper_search_latency = _elapsed_ms(search_started)
        except AudioTranscriptionError:
            LOGGER.warning("Whisper indisponible pour la recherche vocale.")

        # Voice search always obtains one independent, audio-grounded Gemini
        # normalization. Whisper remains the raw transcript and safe fallback.
        trigger_reason = _gemini_trigger_reason(whisper, whisper_response) or "always_normalize"
        if trigger_reason is None:
            if whisper is None or whisper_response is None:
                raise AssistantServiceError("Whisper n'a pas produit de recherche exploitable.")
            transcription = replace(
                whisper,
                whisper_search_latency_ms=whisper_search_latency,
            )
            response = whisper_response
        else:
            transcription, response = self._review_and_compare_searches(
                audio,
                filename,
                content_type,
                top_k,
                whisper,
                whisper_response,
                whisper_search_latency,
                trigger_reason,
            )
        measured_latency_ms = _elapsed_ms(started)
        accounted_latency_ms = (
            sum(
                value
                for key, value in transcription.provider_latencies_ms.items()
                if key in {"whisper", "gemini"}
            )
            + transcription.whisper_search_latency_ms
            + transcription.gemini_search_latency_ms
        )
        latency_ms = max(
            measured_latency_ms,
            accounted_latency_ms,
            transcription.latency_ms
            + transcription.whisper_search_latency_ms
            + transcription.gemini_search_latency_ms,
        )
        LOGGER.info(
            "Recherche vocale terminée: transcription_provider=%s results=%d total_latency_ms=%d",
            transcription.primary_provider,
            len(response.search_result.items),
            latency_ms,
        )
        return AudioAssistantResponse(transcription, response, latency_ms)

    def _review_and_compare_searches(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        top_k: int | None,
        whisper: TranscriptionResult | None,
        whisper_response: AssistantResponse | None,
        whisper_search_latency: int,
        trigger_reason: str,
    ) -> tuple[TranscriptionResult, AssistantResponse]:
        """Review audio once with Gemini and compare unchanged search outcomes."""
        LOGGER.info(
            "Gemini review déclenchée: reason=%s whisper_results=%d",
            trigger_reason,
            len(whisper_response.search_result.items) if whisper_response else 0,
        )
        try:
            gemini = self._transcription_client.review_with_gemini(
                audio, filename, content_type, whisper.text if whisper else None
            )
        except Exception as exc:
            LOGGER.warning("Gemini review indisponible: %s", exc)
            if whisper is None or whisper_response is None:
                raise AssistantServiceError(
                    "Whisper et Gemini n'ont pas produit de transcription exploitable."
                ) from exc
            return (
                replace(
                    whisper,
                    search_recovery_used=True,
                    gemini_triggered=True,
                    gemini_trigger_reason=trigger_reason,
                    whisper_search_latency_ms=whisper_search_latency,
                    resolution_reason="search_recovery_gemini_failed_whisper_preserved",
                ),
                whisper_response,
            )

        gemini_search_started = time.monotonic()
        gemini_response = self.answer(
            gemini.text, top_k, source=SearchSource.AUDIO
        )
        gemini_search_latency = _elapsed_ms(gemini_search_started)
        whisper_result = whisper_response.search_result if whisper_response else SearchResult("", ())
        primary_evidence = _search_evidence(whisper_result)
        alternative_evidence = _search_evidence(gemini_response.search_result)
        evidence = {
            "whisper": _search_evidence_payload(whisper_result),
            "gemini": _search_evidence_payload(gemini_response.search_result),
        }
        LOGGER.info(
            "Voice search recovery comparée: whisper=%s gemini=%s",
            evidence["whisper"],
            evidence["gemini"],
        )
        combined_latencies = dict(whisper.provider_latencies_ms if whisper else {})
        combined_latencies.update(gemini.provider_latencies_ms)
        if alternative_evidence > primary_evidence:
            return (
                replace(
                    gemini,
                    used_fallback=True,
                    whisper_text=whisper.text if whisper else None,
                    gemini_text=gemini.text,
                    alternative_text=whisper.text if whisper else None,
                    search_recovery_used=True,
                    gemini_triggered=True,
                    gemini_trigger_reason=trigger_reason,
                    search_evidence=evidence,
                    resolution_reason="search_recovery_gemini_selected",
                    latency_ms=(
                        (whisper.latency_ms if whisper else 0) + gemini.latency_ms
                    ),
                    provider_latencies_ms=combined_latencies,
                    whisper_search_latency_ms=whisper_search_latency,
                    gemini_search_latency_ms=gemini_search_latency,
                ),
                gemini_response,
            )
        if whisper is None or whisper_response is None:
            return gemini, gemini_response
        return (
            replace(
                whisper,
                gemini_text=gemini.text,
                alternative_text=gemini.text,
                search_recovery_used=True,
                gemini_triggered=True,
                gemini_trigger_reason=trigger_reason,
                search_evidence=evidence,
                resolution_reason="search_recovery_whisper_preserved",
                latency_ms=whisper.latency_ms + gemini.latency_ms,
                provider_latencies_ms=combined_latencies,
                whisper_search_latency_ms=whisper_search_latency,
                gemini_search_latency_ms=gemini_search_latency,
            ),
            whisper_response,
        )


def _search_evidence(result: SearchResult) -> tuple[int, int, int, int, float]:
    """Order outcomes using only existing post-gate evidence."""
    resolved_product_concept = int(
        result.structured_query is not None
        and result.structured_query.product_type is not None
    )
    focal_evidence = sum(_has_transcript_focal_evidence(item, result) for item in result.items)
    best_score = max((item.score for item in result.items), default=float("-inf"))
    primary_count = result.primary_results_count or result.relevant_products_count
    return resolved_product_concept, focal_evidence, primary_count, len(result.items), best_score


def _search_evidence_payload(result: SearchResult) -> dict[str, object]:
    return {
        "relevantProducts": result.relevant_products_count,
        "results": len(result.items),
        "primaryResults": result.primary_results_count,
        "similarResults": result.similar_results_count,
        "matchType": result.match_type,
        "resolvedProductType": (
            result.structured_query.product_type
            if result.structured_query is not None
            else None
        ),
        "bestSemanticScore": (
            max(item.score for item in result.items) if result.items else None
        ),
        "resultIds": [item.product.id for item in result.items],
        "relevanceReasons": [item.relevance_reason for item in result.items],
        "lexicalTerms": [list(item.lexical_terms) for item in result.items],
    }


def _gemini_trigger_reason(
    whisper: TranscriptionResult | None,
    response: AssistantResponse | None,
) -> str | None:
    if whisper is None:
        return "whisper_failed"
    if whisper.metadata.get("is_reliable") is False:
        return "low_quality"
    if response is None:
        return "search_no_results"
    result = response.search_result
    if not result.items:
        return "search_no_results"
    if result.match_type == "similar":
        return "weak_search_evidence"
    if (
        result.match_type != "none"
        and result.primary_results_count == 0
    ):
        return "weak_search_evidence"
    reasons = tuple(item.relevance_reason for item in result.items)
    has_strong_focal_evidence = any(
        _has_transcript_focal_evidence(item, result) for item in result.items
    )
    weak_reasons = {"isolated_semantic_leader", "focal_lexical_evidence"}
    # In production SearchService exposes the resolved product concept. For an
    # audio transcript, a clean lexical overlap can still be a confidently
    # misheard generic word. Review once when no product concept was resolved;
    # the second hypothesis still has to win through the same search pipeline.
    if (
        result.structured_query is not None
        and result.structured_query.product_type is None
        and reasons
        and all(reason in weak_reasons for reason in reasons)
    ):
        return "weak_search_evidence"
    if (
        reasons
        and not has_strong_focal_evidence
        and all(reason in weak_reasons for reason in reasons)
    ):
        return "weak_search_evidence"
    return None


def _has_transcript_focal_evidence(item: object, result: SearchResult) -> bool:
    """Require focal terms to be grounded in the actual audio transcript.

    Empty lexical metadata is treated as legacy evidence for compatibility;
    production SearchService always supplies the aligned gate terms.
    """
    if getattr(item, "relevance_reason", None) != "focal_lexical_evidence":
        return False
    lexical_terms = tuple(getattr(item, "lexical_terms", ()))
    if not lexical_terms:
        return True
    original = (
        result.structured_query.original_query
        if result.structured_query is not None
        else result.query
    )
    original_tokens = set(
        re.findall(r"[^\W_]+", _normalise_text(original), flags=re.UNICODE)
    )
    return any(_normalise_text(term) in original_tokens for term in lexical_terms)


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
