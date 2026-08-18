from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock

from config import Settings
from embeddings.embedder import EmbeddingService
from models.product import Product


def create_product() -> Product:
    return Product(
        id="123",
        title="Laptop Dell XPS",
        description="Ordinateur portable Dell XPS",
        category="Électronique",
        brand="Dell",
        color="Noir",
        condition="Très bon état",
        price=Decimal("400"),
        currency="MAD",
        city="Casablanca",
        image_urls=(),
        status="ACCEPTED",
        is_sold=False,
        updated_at=datetime.now(timezone.utc),
    )


def create_settings() -> Settings:
    return Settings(
        environment="test",
        host="127.0.0.1",
        port=5000,
        log_level="WARNING",
        embedding_model_name="intfloat/multilingual-e5-base",
        embedding_device="cpu",
    )


def test_product_embedding_uses_passage_prefix():
    model = Mock()

    model.encode.return_value = [
        [1.0, 0.0, 0.0]
    ]

    service = EmbeddingService(
        create_settings(),
        model=model,
    )

    service.embed_product(create_product())

    model.encode.assert_called_once()

    texts = model.encode.call_args.args[0]

    assert len(texts) == 1
    assert texts[0].startswith("passage: ")


def test_query_embedding_uses_query_prefix():
    model = Mock()

    model.encode.return_value = [
        [1.0, 0.0, 0.0]
    ]

    service = EmbeddingService(
        create_settings(),
        model=model,
    )

    service.embed_query("ordinateur portable")

    model.encode.assert_called_once()

    texts = model.encode.call_args.args[0]

    assert len(texts) == 1
    assert texts[0] == "query: ordinateur portable"
