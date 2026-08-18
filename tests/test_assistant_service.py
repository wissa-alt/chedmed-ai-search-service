"""Unit tests for catalogue answer and transcription orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from models.product import Product
from models.query_analysis import QueryIntent, SupportedLanguage
from models.search_query import SearchSource, StructuredSearchQuery
from models.transcription import TranscriptionResult
from search.search_service import SearchResult, SearchResultItem
from services.assistant_service import AssistantService, AssistantServiceError


@pytest.fixture
def product() -> Product:
    """Return a product used as answer evidence."""
    return Product(
        id="product-1", title="Vélo", description="Description", category="Vélos",
        brand=None, color=None, condition=None, price=Decimal("100"), currency="MAD",
        city=None, image_urls=(), status="ACTIVE", is_sold=False,
        updated_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )


def test_answer_grounds_llm_call_in_search_results(product: Product) -> None:
    """The answer client receives only products returned by semantic search."""
    search = Mock()
    result = SearchResult("vélo", (SearchResultItem(product, 0.8),))
    search.search.return_value = result
    answer_client = Mock()
    answer_client.generate_catalogue_answer.return_value = "Voici un vélo."
    service = AssistantService(search, answer_client, Mock())

    response = service.answer("vélo", 3)

    assert response.answer == "Voici un vélo."
    assert response.search_result == result
    answer_client.generate_catalogue_answer.assert_called_once_with("vélo", (product,))


def test_answer_labels_fallback_products_as_similar(product: Product) -> None:
    search = Mock()
    search.search.return_value = SearchResult(
        "shirt",
        (SearchResultItem(product, 0.8, match_type="similar"),),
        similar_results_count=1,
        match_type="similar",
    )
    answer_client = Mock()
    answer_client.generate_catalogue_answer.return_value = "Voici le produit proche."

    response = AssistantService(search, answer_client, Mock()).answer("shirt")

    assert response.answer.startswith(
        "Aucune correspondance suffisamment précise n'a été trouvée."
    )
    assert response.answer.endswith("Voici le produit proche.")


def test_answer_translates_collaborator_failure() -> None:
    """Search or provider failures do not leak through the service boundary."""
    search = Mock()
    search.search.side_effect = RuntimeError("unavailable")
    service = AssistantService(search, Mock(), Mock())

    with pytest.raises(AssistantServiceError):
        service.answer("vélo")


def test_empty_search_uses_fixed_grounded_answer_without_llm() -> None:
    search = Mock()
    search.search.return_value = SearchResult("mot inconnu", ())
    answer_client = Mock()
    response = AssistantService(search, answer_client, Mock()).answer("mot inconnu")
    assert response.answer == (
        "Aucun produit suffisamment pertinent ou similaire n'a été trouvé."
    )
    answer_client.generate_catalogue_answer.assert_not_called()


def test_transcribe_delegates_in_memory_bytes() -> None:
    """No upload persistence is needed for voice transcription."""
    transcription = Mock()
    result = TranscriptionResult("bonjour", "whisper", False)
    transcription.transcribe.return_value = result
    service = AssistantService(Mock(), Mock(), transcription)

    assert service.transcribe(b"audio", "message.wav", "audio/wav") == result
    transcription.transcribe.assert_called_once_with(b"audio", "message.wav", "audio/wav")


def test_answer_audio_always_normalizes_then_reuses_text_pipeline(product: Product) -> None:
    """Voice search always obtains Gemini text and uses the established search."""
    search = Mock()
    search_result = SearchResult(
        "gaming laptop",
        (SearchResultItem(product, 0.8, "focal_lexical_evidence"),),
        relevant_products_count=1,
    )
    search.search.return_value = search_result
    answer_client = Mock()
    answer_client.generate_catalogue_answer.return_value = "Résultat"
    transcription = Mock()
    transcription_result = TranscriptionResult("gaming laptop", "whisper", False)
    transcription.transcribe_whisper_primary.return_value = transcription_result
    transcription.review_with_gemini.return_value = TranscriptionResult(
        "gaming laptop", "gemini", False
    )
    response = AssistantService(search, answer_client, transcription).answer_audio(
        b"audio", "voice.wav", "audio/wav", 5
    )
    assert response.transcription.gemini_triggered is True
    assert response.transcription.gemini_trigger_reason == "always_normalize"
    assert response.transcription.gemini_text == "gaming laptop"
    assert response.assistant_response.search_result == search_result
    assert search.search.call_count == 2
    transcription.review_with_gemini.assert_called_once_with(
        b"audio", "voice.wav", "audio/wav", "gaming laptop"
    )


def test_high_quality_weak_semantic_result_triggers_gemini_and_selects_focal_result(
    product: Product,
) -> None:
    """Existing RelevanceGate reasons drive recovery without a new score."""
    search = Mock()
    weak = SearchResult(
        "misheard words",
        (SearchResultItem(product, 0.91, "isolated_semantic_leader"),),
        relevant_products_count=1,
    )
    focal = SearchResult(
        "correct words",
        (SearchResultItem(product, 0.84, "focal_lexical_evidence"),),
        relevant_products_count=1,
    )
    search.search.side_effect = [weak, focal]
    answer_client = Mock()
    answer_client.generate_catalogue_answer.return_value = "Produit trouvé"
    audio_service = Mock()
    audio_service.transcribe_whisper_primary.return_value = TranscriptionResult(
        "misheard words",
        "whisper",
        False,
        quality_score=1.0,
        metadata={"is_reliable": True},
        provider_latencies_ms={"whisper": 100},
    )
    audio_service.review_with_gemini.return_value = TranscriptionResult(
        "correct words",
        "gemini",
        False,
        provider_latencies_ms={"gemini": 200},
    )

    response = AssistantService(search, answer_client, audio_service).answer_audio(
        b"same-audio", "voice.wav", "audio/wav", 5
    )

    audio_service.review_with_gemini.assert_called_once_with(
        b"same-audio", "voice.wav", "audio/wav", "misheard words"
    )
    assert response.transcription.text == "correct words"
    assert response.transcription.alternative_text == "misheard words"
    assert response.transcription.gemini_trigger_reason == "weak_search_evidence"
    assert response.transcription.resolution_reason == "search_recovery_gemini_selected"
    assert response.transcription.search_evidence["whisper"]["relevanceReasons"] == [
        "isolated_semantic_leader"
    ]
    assert response.transcription.search_evidence["gemini"]["relevanceReasons"] == [
        "focal_lexical_evidence"
    ]
    assert response.latency_ms >= 300


def test_weak_gemini_outcome_does_not_replace_better_whisper_result(
    product: Product,
) -> None:
    search = Mock()
    whisper_search = SearchResult(
        "primary words",
        (SearchResultItem(product, 0.91, "isolated_semantic_leader"),),
        relevant_products_count=1,
    )
    gemini_search = SearchResult(
        "alternative words",
        (SearchResultItem(product, 0.82, "isolated_semantic_leader"),),
        relevant_products_count=1,
    )
    search.search.side_effect = [whisper_search, gemini_search]
    audio_service = Mock()
    audio_service.transcribe_whisper_primary.return_value = TranscriptionResult(
        "primary words", "whisper", False, metadata={"is_reliable": True}
    )
    audio_service.review_with_gemini.return_value = TranscriptionResult(
        "alternative words", "gemini", False
    )

    response = AssistantService(search, Mock(), audio_service).answer_audio(
        b"audio", "voice.wav", "audio/wav"
    )

    assert response.transcription.text == "primary words"
    assert response.transcription.alternative_text == "alternative words"
    assert response.transcription.gemini_trigger_reason == "weak_search_evidence"
    assert response.transcription.resolution_reason == "search_recovery_whisper_preserved"


def test_audio_recovery_prefers_resolved_product_concept_over_weak_primary(
    product: Product,
) -> None:
    """A normalized concept may select shared family fallback over a misheard leader."""
    search = Mock()
    weak = SearchResult(
        "misheard speech",
        (SearchResultItem(product, 0.91, "focal_lexical_evidence", ("speech",)),),
        structured_query=StructuredSearchQuery(
            original_query="misheard speech",
            semantic_query="misheard speech",
            language=SupportedLanguage.ENGLISH,
            intent=QueryIntent.PRODUCT_SEARCH,
            source=SearchSource.AUDIO,
        ),
        relevant_products_count=1,
        primary_results_count=1,
        match_type="relevant",
    )
    resolved = StructuredSearchQuery(
        original_query="normalized request",
        semantic_query="normalized request",
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        source=SearchSource.AUDIO,
        product_type="chaussures",
    )
    recovered = SearchResult(
        "normalized request",
        (SearchResultItem(product, 0.84, "product_family:footwear", (), "similar"),),
        structured_query=resolved,
        similar_results_count=1,
        match_type="similar",
    )
    search.search.side_effect = [weak, recovered]
    audio_service = Mock()
    audio_service.transcribe_whisper_primary.return_value = TranscriptionResult(
        "misheard speech", "whisper", False, metadata={"is_reliable": True}
    )
    audio_service.review_with_gemini.return_value = TranscriptionResult(
        "normalized request", "gemini", False
    )
    answer_client = Mock()
    answer_client.generate_catalogue_answer.return_value = "Résultat catalogue"

    response = AssistantService(search, answer_client, audio_service).answer_audio(
        b"same-audio", "voice.wav", "audio/wav"
    )

    assert response.transcription.text == "normalized request"
    assert response.transcription.resolution_reason == "search_recovery_gemini_selected"
    assert response.transcription.search_evidence["gemini"]["resolvedProductType"] == "chaussures"
    audio_service.review_with_gemini.assert_called_once_with(
        b"same-audio", "voice.wav", "audio/wav", "misheard speech"
    )


def test_empty_whisper_search_retries_same_audio_with_gemini_and_selects_results(
    product: Product,
) -> None:
    """Search recovery compares two complete text-pipeline outcomes."""
    search = Mock()
    whisper_search = SearchResult(
        "gaming lab top", (), faiss_candidates_count=20, relevant_products_count=0
    )
    gemini_search = SearchResult(
        "gaming laptop",
        (SearchResultItem(product, 0.86),),
        faiss_candidates_count=20,
        filtered_products_count=1,
        relevant_products_count=1,
    )
    search.search.side_effect = [whisper_search, gemini_search]
    answer_client = Mock()
    answer_client.generate_catalogue_answer.return_value = "Voici le produit Vélo."
    audio_service = Mock()
    audio_service.transcribe_whisper_primary.return_value = TranscriptionResult(
        "gaming lab top",
        "whisper",
        False,
        quality_score=0.91,
        latency_ms=100,
        metadata={"is_reliable": True},
    )
    audio_service.review_with_gemini.return_value = TranscriptionResult(
        "gaming laptop",
        "gemini",
        False,
        quality_score=1.0,
        latency_ms=250,
        provider_latencies_ms={"gemini": 250},
    )
    response = AssistantService(search, answer_client, audio_service).answer_audio(
        b"same-original-audio", "voice.wav", "audio/wav", 5
    )

    assert response.transcription.text == "gaming laptop"
    assert response.transcription.alternative_text == "gaming lab top"
    assert response.transcription.whisper_text == "gaming lab top"
    assert response.transcription.gemini_text == "gaming laptop"
    assert response.transcription.resolution_reason == "search_recovery_gemini_selected"
    assert response.transcription.search_recovery_used is True
    assert response.assistant_response.search_result == gemini_search
    assert response.assistant_response.answer == "Voici le produit Vélo."
    assert response.latency_ms >= 350
    audio_service.review_with_gemini.assert_called_once_with(
        b"same-original-audio", "voice.wav", "audio/wav", "gaming lab top"
    )
    assert [entry.args for entry in search.search.call_args_list] == [
        ("gaming lab top", 5),
        ("gaming laptop", 5),
    ]
    assert all(
        entry.kwargs == {"source": SearchSource.AUDIO}
        for entry in search.search.call_args_list
    )


def test_empty_alternative_search_preserves_whisper_and_both_hypotheses() -> None:
    search = Mock()
    search.search.side_effect = [
        SearchResult("primary words", ()),
        SearchResult("alternative words", ()),
    ]
    answer_client = Mock()
    answer_client.generate_catalogue_answer.return_value = "Aucun résultat."
    audio_service = Mock()
    audio_service.transcribe_whisper_primary.return_value = TranscriptionResult(
        "primary words", "whisper", False, metadata={"is_reliable": True}
    )
    audio_service.review_with_gemini.return_value = TranscriptionResult(
        "alternative words", "gemini", False
    )
    response = AssistantService(search, answer_client, audio_service).answer_audio(
        b"audio", "voice.wav", "audio/wav"
    )
    assert response.transcription.text == "primary words"
    assert response.transcription.alternative_text == "alternative words"
    assert response.transcription.resolution_reason == "search_recovery_whisper_preserved"
    assert response.assistant_response.search_result.items == ()


def test_whisper_exception_triggers_gemini_with_same_audio_and_no_text(
    product: Product,
) -> None:
    search = Mock()
    gemini_result = SearchResult(
        "corrected words",
        (SearchResultItem(product, 0.88),),
        relevant_products_count=1,
    )
    search.search.return_value = gemini_result
    answer_client = Mock()
    answer_client.generate_catalogue_answer.return_value = "Vélo trouvé"
    audio_service = Mock()
    from services.audio_errors import AudioProviderError

    audio_service.transcribe_whisper_primary.side_effect = AudioProviderError("down")
    audio_service.review_with_gemini.return_value = TranscriptionResult(
        "corrected words", "gemini", True
    )
    response = AssistantService(search, answer_client, audio_service).answer_audio(
        b"same-audio", "voice.wav", "audio/wav"
    )
    audio_service.review_with_gemini.assert_called_once_with(
        b"same-audio", "voice.wav", "audio/wav", None
    )
    assert response.transcription.gemini_trigger_reason == "whisper_failed"
    assert response.transcription.text == "corrected words"
    search.search.assert_called_once_with(
        "corrected words", None, source=SearchSource.AUDIO
    )


def test_gemini_recovery_failure_preserves_primary_empty_outcome() -> None:
    search = Mock()
    primary_result = SearchResult("primary words", ())
    search.search.return_value = primary_result
    answer_client = Mock()
    answer_client.generate_catalogue_answer.return_value = "Aucun résultat."
    audio_service = Mock()
    audio_service.transcribe_whisper_primary.return_value = TranscriptionResult(
        "primary words", "whisper", False, metadata={"is_reliable": True}
    )
    audio_service.review_with_gemini.side_effect = RuntimeError("Gemini down")
    response = AssistantService(search, answer_client, audio_service).answer_audio(
        b"audio", "voice.wav", "audio/wav"
    )
    assert response.transcription.text == "primary words"
    assert response.transcription.resolution_reason == (
        "search_recovery_gemini_failed_whisper_preserved"
    )
    search.search.assert_called_once_with(
        "primary words", None, source=SearchSource.AUDIO
    )


def test_low_quality_whisper_triggers_one_gemini_review(product: Product) -> None:
    search = Mock()
    search.search.side_effect = [
        SearchResult("primary words", ()),
        SearchResult(
            "alternative words",
            (SearchResultItem(product, 0.82),),
            relevant_products_count=1,
        ),
    ]
    answer_client = Mock()
    answer_client.generate_catalogue_answer.side_effect = ["Aucun", "Vélo trouvé"]
    audio_service = Mock()
    audio_service.transcribe_whisper_primary.return_value = TranscriptionResult(
        "primary words",
        "whisper",
        False,
        quality_score=0.4,
        metadata={"is_reliable": False},
    )
    audio_service.review_with_gemini.return_value = TranscriptionResult(
        "alternative words",
        "gemini",
        True,
        quality_score=1.0,
    )
    response = AssistantService(search, answer_client, audio_service).answer_audio(
        b"audio", "voice.wav", "audio/wav"
    )
    assert response.transcription.text == "alternative words"
    assert response.transcription.quality_score == 1.0
    assert response.transcription.gemini_trigger_reason == "low_quality"
    audio_service.review_with_gemini.assert_called_once_with(
        b"audio", "voice.wav", "audio/wav", "primary words"
    )
