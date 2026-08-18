"""Unit tests for the injected Groq adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest

from config import Settings
from llm.groq_client import GroqClient, GroqClientError, GroqTranscription
from models.product import Product


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return settings with non-sensitive test provider configuration."""
    return Settings(
        environment="test", host="127.0.0.1", port=5000, log_level="CRITICAL",
        db_host="127.0.0.1", db_port=5432, db_name="chedmed", db_user="test", db_password="password",
        chedmed_webhook_secret="secret", groq_api_key="groq", project_root=tmp_path,
    )


@pytest.fixture
def product() -> Product:
    """Return one product used to build a grounded prompt."""
    return Product(
        id="product-1", title="Vélo", description="Description", category="Vélos",
        brand=None, color=None, condition=None, price=Decimal("100"), currency="MAD",
        city=None, image_urls=(), status="ACTIVE", is_sold=False,
        updated_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )


def test_generate_catalogue_answer_uses_injected_sdk(settings: Settings, product: Product) -> None:
    """The client uses the configured chat model and returns text content."""
    sdk = Mock()
    sdk.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Réponse pour Vélo product-1"))]
    )
    client = GroqClient(settings, sdk)

    assert client.generate_catalogue_answer("vélo", [product]) == "Réponse pour Vélo product-1"
    call = sdk.chat.completions.create.call_args
    assert call.kwargs["model"] == settings.groq_chat_model
    assert "product-1" in call.kwargs["messages"][1]["content"]


def test_transcribe_uses_in_memory_file(settings: Settings) -> None:
    """Audio bytes are sent to the injected SDK without any disk write."""
    sdk = Mock()
    sdk.audio.transcriptions.create.return_value = {
        "text": "bonjour",
        "language": "fr",
        "duration": 1.2,
        "segments": [
            {"text": "bonjour", "avg_logprob": -0.2, "no_speech_prob": 0.01}
        ],
    }
    client = GroqClient(settings, sdk)

    result = client.transcribe(b"audio", "clip.wav", "audio/wav")
    assert result == GroqTranscription(
        text="bonjour",
        language="fr",
        duration=1.2,
        segments=({"text": "bonjour", "avg_logprob": -0.2, "no_speech_prob": 0.01},),
    )
    call = sdk.audio.transcriptions.create.call_args.kwargs
    assert call["model"] == settings.groq_whisper_model
    assert call["response_format"] == "verbose_json"


def test_empty_provider_response_is_rejected(settings: Settings) -> None:
    """An empty provider completion cannot become a successful application answer."""
    sdk = Mock()
    sdk.chat.completions.create.return_value = Mock(choices=[Mock(message=Mock(content=" "))])
    client = GroqClient(settings, sdk)

    with pytest.raises(GroqClientError, match="vide"):
        client.generate_catalogue_answer("vélo", [])


def test_seller_description_prompt_forbids_missing_attributes(settings: Settings) -> None:
    sdk = Mock()
    sdk.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Dell XPS noir en très bon état."))]
    )
    result = GroqClient(settings, sdk).suggest_seller_description(
        {
            "title": "Dell XPS",
            "description": "Laptop Dell bon état",
            "category": "Électronique",
            "brand": "Dell",
            "color": "Noir",
            "condition": "Très bon état",
        }
    )
    prompt = sdk.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert result == "Dell XPS noir en très bon état."
    assert "UNIQUEMENT" in prompt
    assert "RAM" in prompt
    assert "prix" in prompt
    assert "sellerPrice" not in prompt


def test_contradictory_no_product_answer_is_replaced_with_grounded_result(
    settings: Settings, product: Product
) -> None:
    """A non-empty result list cannot be described as having no product."""
    sdk = Mock()
    sdk.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Aucun produit ne correspond."))]
    )
    answer = GroqClient(settings, sdk).generate_catalogue_answer("vélo", [product])
    assert "Vélo" in answer
    assert "product-1" in answer
    assert "Aucun produit" not in answer
