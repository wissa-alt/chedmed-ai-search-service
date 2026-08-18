"""Unit tests for the dependency-free embedding service boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from config import Settings
from embeddings.embedder import (
    EmbeddingService,
    EmbeddingServiceError,
    _resolve_device,
)
from models.product import Product


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Build complete test settings without reading process configuration."""
    return Settings(
        environment="test",
        host="127.0.0.1",
        port=5000,
        log_level="CRITICAL",
        db_host="127.0.0.1", db_port=5432, db_name="chedmed", db_user="test", db_password="password",
        chedmed_webhook_secret="test-webhook-secret",
        groq_api_key="test-groq-key",
        project_root=tmp_path,
    )


@pytest.fixture
def product() -> Product:
    """Return a representative product without requiring the API client."""
    return Product(
        id="not-indexed-as-text",
        title="Vélo électrique",
        description="Autonomie de 70 km.",
        category="Vélos",
        brand="ChedMed",
        color="Noir",
        condition="Très bon état",
        price=Decimal("1250.00"),
        currency="MAD",
        city="Casablanca",
        image_urls=("https://images.test/velo.jpg",),
        status="ACTIVE",
        is_sold=False,
        updated_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
    )


@pytest.fixture
def model() -> Mock:
    """Return a deterministic SentenceTransformer test double."""
    fake_model = Mock()
    fake_model.encode.side_effect = lambda texts, **_: np.tile(
        np.array([3.0, 4.0], dtype=np.float32), (len(texts), 1)
    )
    return fake_model


@pytest.fixture
def service(settings: Settings, model: Mock) -> EmbeddingService:
    """Create an in-process service double without downloading a model."""
    return EmbeddingService(settings, model=model)


def test_embed_product_returns_normalised_vector(
    service: EmbeddingService, model: Mock, product: Product
) -> None:
    """One product is converted to a normalised E5 passage embedding."""
    vector = service.embed_product(product)

    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0)
    encoded_document = model.encode.call_args.args[0][0]
    assert encoded_document.startswith("passage: ")
    assert "Titre: Vélo électrique" in encoded_document
    assert "not-indexed-as-text" not in encoded_document
    assert "images.test" not in encoded_document


def test_embed_products_uses_one_batch_call(
    service: EmbeddingService, model: Mock, product: Product
) -> None:
    """Multiple products are passed to SentenceTransformer as one batch."""
    second_product = replace(product, id="product-2", title="Casque vélo")

    vectors = service.embed_products([product, second_product])

    assert len(vectors) == 2
    assert all(np.isclose(np.linalg.norm(vector), 1.0) for vector in vectors)
    assert model.encode.call_count == 1
    assert len(model.encode.call_args.args[0]) == 2


def test_embed_query_uses_e5_query_prefix(service: EmbeddingService, model: Mock) -> None:
    """User queries use E5's distinct query prefix automatically."""
    vector = service.embed_query("vélo électrique noir")

    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert model.encode.call_args.args[0] == ["query: vélo électrique noir"]


def test_product_without_description_is_still_indexable(
    service: EmbeddingService, model: Mock, product: Product
) -> None:
    """An optional blank description is omitted instead of blocking indexing."""
    product_without_description = replace(product, description="   ")

    service.embed_product(product_without_description)

    document = model.encode.call_args.args[0][0]
    assert "Description:" not in document
    assert "Titre: Vélo électrique" in document


def test_missing_optional_fields_are_ignored(
    service: EmbeddingService, model: Mock, product: Product
) -> None:
    """Absent optional display fields do not appear in the semantic document."""
    sparse_product = replace(
        product,
        brand=None,
        color=None,
        condition=None,
        city=None,
    )

    service.embed_product(sparse_product)

    document = model.encode.call_args.args[0][0]
    assert "Marque:" not in document
    assert "Couleur:" not in document
    assert "État:" not in document
    assert "Ville:" not in document


def test_model_encoding_error_becomes_domain_error(
    service: EmbeddingService, model: Mock, product: Product
) -> None:
    """SentenceTransformer failures do not leak provider-specific exceptions."""
    model.encode.side_effect = RuntimeError("model unavailable")

    with pytest.raises(EmbeddingServiceError, match="SentenceTransformers"):
        service.embed_product(product)


def test_model_loading_error_becomes_domain_error(settings: Settings, mocker: pytest.MockFixture) -> None:
    """A failed isolated runtime becomes an explicit service exception."""
    mocker.patch("embeddings.isolated_runtime.IsolatedEmbeddingRuntime", side_effect=OSError("worker unavailable"))

    with pytest.raises(EmbeddingServiceError, match="modèle d'embeddings est indisponible"):
        EmbeddingService(settings)


def test_production_service_creates_one_isolated_runtime(
    settings: Settings,
    mocker: pytest.MockFixture,
) -> None:
    """The production path owns exactly one FAISS-free model runtime."""
    runtime = Mock()
    runtime_factory = mocker.patch(
        "embeddings.isolated_runtime.IsolatedEmbeddingRuntime",
        return_value=runtime,
    )

    service = EmbeddingService(settings)
    service.close()

    runtime_factory.assert_called_once_with(
        settings.embedding_model_name,
        "cpu",
    )
    runtime.close.assert_called_once_with()


def test_auto_device_uses_cpu_without_importing_torch() -> None:
    """Automatic selection cannot initialise PyTorch in the FAISS process."""
    assert _resolve_device("auto") == "cpu"
