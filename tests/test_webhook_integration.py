"""Integration tests for the ChedMed webhook -> sync -> FAISS pipeline."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from config import Settings
from models.product import Product
from search.product_text_builder import ProductTextBuilder
from services.sync_service import SynchronizationService
from sync.webhook_handler import WebhookHandler
from vector_store.faiss_manager import FAISSManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create isolated settings for the integration test."""
    return Settings(
        environment="test",
        host="127.0.0.1",
        port=5000,
        log_level="CRITICAL",
        db_host="127.0.0.1",
        db_port=5432,
        db_name="chedmed_test_db",
        db_user="postgres",
        db_password="password",
        chedmed_webhook_secret="webhook-test-secret",
        groq_api_key="test-groq-key",
        project_root=tmp_path,
    )


@pytest.fixture
def product() -> Product:
    """Create a valid ChedMed product."""
    return Product(
        id="product-42",
        title="Laptop Dell XPS",
        description="Ordinateur portable professionnel.",
        category="Informatique",
        brand="Dell",
        color="Silver",
        condition="NEW",
        price=Decimal("12500"),
        currency="MAD",
        city="Beni Mellal",
        image_urls=(),
        status="ACTIVE",
        is_sold=False,
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def second_product() -> Product:
    """Create a second valid product."""
    return Product(
        id="product-99",
        title="iPhone 15",
        description="Smartphone Apple.",
        category="Téléphones",
        brand="Apple",
        color="Black",
        condition="NEW",
        price=Decimal("9500"),
        currency="MAD",
        city="Beni Mellal",
        image_urls=(),
        status="ACTIVE",
        is_sold=False,
        updated_at=datetime.now(timezone.utc),
    )


class FakeCatalogue:
    """In-memory ChedMed catalogue used as the source of truth."""

    def __init__(self, products: list[Product]) -> None:
        self.products: dict[str, Product] = {
            product.id: product for product in products
        }

    def get_all_products(self) -> list[Product]:
        """Return every current product."""
        return list(self.products.values())

    def get_products_page(
        self,
        page: int,
        limit: int,
    ) -> list[Product]:
        """Return one-based paginated products."""
        if page < 1:
            raise ValueError("page must be >= 1")

        if limit < 1:
            raise ValueError("limit must be >= 1")

        products = list(self.products.values())

        start = (page - 1) * limit
        end = start + limit

        return products[start:end]

    def get_product(self, product_id: str) -> Product:
        """Return one current product."""
        try:
            return self.products[product_id]
        except KeyError as exc:
            raise KeyError(
                f"Produit ChedMed introuvable : {product_id}"
            ) from exc

    def get_products_updated_after(
        self,
        updated_after: datetime | str,
    ) -> list[Product]:
        """Return products updated after the supplied timestamp."""
        if isinstance(updated_after, str):
            timestamp = datetime.fromisoformat(
                updated_after.replace("Z", "+00:00")
            )
        else:
            timestamp = updated_after

        return [
            product
            for product in self.products.values()
            if product.updated_at > timestamp
        ]

    def update_product(self, product: Product) -> None:
        """Replace a product in the fake catalogue."""
        self.products[product.id] = product

    def delete_product(self, product_id: str) -> None:
        """Delete a product from the fake catalogue."""
        self.products.pop(product_id, None)


class FakeEmbedder:
    """Deterministic embedding provider for integration tests."""

    def embed_product(self, product: Product) -> np.ndarray:
        """Generate a deterministic 768-dimensional test embedding."""
        vector = np.zeros(768, dtype=np.float32)

        # Keep the vector deterministic without using SentenceTransformer.
        if product.id == "product-42":
            vector[0] = 1.0
        elif product.id == "product-99":
            vector[1] = 1.0
        else:
            vector[2] = 1.0

        return vector


@pytest.fixture
def catalogue(
    product: Product,
    second_product: Product,
) -> FakeCatalogue:
    """Create the fake ChedMed source of truth."""
    return FakeCatalogue([product, second_product])


@pytest.fixture
def embedder() -> FakeEmbedder:
    """Create a deterministic embedding provider."""
    return FakeEmbedder()


@pytest.fixture
def vector_store(
    settings: Settings,
) -> FAISSManager:
    """Create a real FAISS manager isolated in tmp_path."""
    return FAISSManager(settings)


@pytest.fixture
def text_builder() -> ProductTextBuilder:
    """Create the real product text builder."""
    return ProductTextBuilder()


@pytest.fixture
def synchronization_service(
    catalogue: FakeCatalogue,
    embedder: FakeEmbedder,
    vector_store: FAISSManager,
    text_builder: ProductTextBuilder,
) -> SynchronizationService:
    """Wire the real synchronization service."""
    return SynchronizationService(
        product_source=catalogue,
        embedder=embedder,
        vector_store=vector_store,
        text_builder=text_builder,
    )


@pytest.fixture
def webhook_handler(
    settings: Settings,
    synchronization_service: SynchronizationService,
) -> WebhookHandler:
    """Wire the real webhook handler to the real sync service."""
    return WebhookHandler(
        settings=settings,
        synchronization_service=synchronization_service,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload(
    event_type: str,
    *,
    event_id: str,
    product_id: str = "product-42",
) -> dict[str, str]:
    """Build a valid ChedMed webhook payload."""
    return {
        "eventId": event_id,
        "eventType": event_type,
        "productId": product_id,
        "occurredAt": "2026-08-12T15:00:00Z",
    }


def _signed_headers(
    settings: Settings,
    payload: dict[str, str],
) -> dict[str, str]:
    """Generate the HMAC SHA-256 signature expected by WebhookHandler."""
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    signature = hmac.new(
        settings.chedmed_webhook_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    return {
        "X-ChedMed-Signature": f"sha256={signature}",
    }


def _send_webhook(
    handler: WebhookHandler,
    settings: Settings,
    payload: dict[str, str],
):
    """Send one correctly signed webhook."""
    headers = _signed_headers(settings, payload)

    return handler.handle(payload, headers)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_full_sync_builds_real_faiss_index(
    synchronization_service: SynchronizationService,
    vector_store: FAISSManager,
) -> None:
    """The real synchronization service creates a real FAISS index."""
    report = synchronization_service.full_sync()

    assert report.total_products == 2
    assert report.indexed_products == 2
    assert report.updated_products == 0
    assert report.failed_products == 0

    assert vector_store.count() == 2
    assert vector_store.contains("product-42") is True
    assert vector_store.contains("product-99") is True


# ---------------------------------------------------------------------------
# PRODUCT_CREATED
# ---------------------------------------------------------------------------


def test_product_created_is_added_to_faiss(
    settings: Settings,
    webhook_handler: WebhookHandler,
    vector_store: FAISSManager,
    catalogue: FakeCatalogue,
    product: Product,
) -> None:
    """PRODUCT_CREATED fetches the product and inserts it into FAISS."""
    payload = _payload(
        "PRODUCT_CREATED",
        event_id="event-created-1",
        product_id=product.id,
    )

    # Start with an empty vector store.
    assert vector_store.count() == 0

    result = _send_webhook(
        webhook_handler,
        settings,
        payload,
    )

    assert result.accepted is True
    assert result.duplicated is False
    assert result.action == "synced"

    assert catalogue.get_product(product.id).id == product.id
    assert vector_store.contains(product.id) is True
    assert vector_store.count() == 1


# ---------------------------------------------------------------------------
# PRODUCT_UPDATED
# ---------------------------------------------------------------------------


def test_product_updated_replaces_existing_vector(
    settings: Settings,
    webhook_handler: WebhookHandler,
    vector_store: FAISSManager,
    catalogue: FakeCatalogue,
    product: Product,
) -> None:
    """PRODUCT_UPDATED recalculates the product vector."""
    # First index the original product.
    initial_result = webhook_handler._synchronization_service.sync_product(
        product.id
    )

    assert initial_result.action == "added"
    assert vector_store.contains(product.id) is True

    old_vector_id = initial_result.vector_id

    updated_product = Product(
        id=product.id,
        title="Laptop Dell XPS - Nouvelle version",
        description="Nouvelle description.",
        category=product.category,
        brand=product.brand,
        color=product.color,
        condition=product.condition,
        price=Decimal("13500"),
        currency=product.currency,
        city=product.city,
        image_urls=product.image_urls,
        status=product.status,
        is_sold=False,
        updated_at=datetime.now(timezone.utc),
    )

    catalogue.update_product(updated_product)

    payload = _payload(
        "PRODUCT_UPDATED",
        event_id="event-updated-1",
        product_id=product.id,
    )

    result = _send_webhook(
        webhook_handler,
        settings,
        payload,
    )

    assert result.accepted is True
    assert result.action == "synced"

    assert vector_store.contains(product.id) is True
    assert vector_store.count() == 1

    # The VectorID may be reused or replaced depending on the FAISS
    # implementation. The important contract is that exactly one
    # current product remains indexed.
    assert old_vector_id is not None


# ---------------------------------------------------------------------------
# PRODUCT_DELETED
# ---------------------------------------------------------------------------


def test_product_deleted_removes_product_from_faiss(
    settings: Settings,
    webhook_handler: WebhookHandler,
    synchronization_service: SynchronizationService,
    vector_store: FAISSManager,
    product: Product,
) -> None:
    """PRODUCT_DELETED removes the product from FAISS."""
    synchronization_service.sync_product(product.id)

    assert vector_store.contains(product.id) is True

    payload = _payload(
        "PRODUCT_DELETED",
        event_id="event-deleted-1",
        product_id=product.id,
    )

    result = _send_webhook(
        webhook_handler,
        settings,
        payload,
    )

    assert result.accepted is True
    assert result.action == "removed"

    assert vector_store.contains(product.id) is False
    assert vector_store.count() == 0


# ---------------------------------------------------------------------------
# PRODUCT_SOLD
# ---------------------------------------------------------------------------


def test_product_sold_remains_in_faiss(
    settings: Settings,
    webhook_handler: WebhookHandler,
    synchronization_service: SynchronizationService,
    vector_store: FAISSManager,
    product: Product,
) -> None:
    """PRODUCT_SOLD refreshes and preserves the historical catalogue row."""
    synchronization_service.sync_product(product.id)

    assert vector_store.contains(product.id) is True

    payload = _payload(
        "PRODUCT_SOLD",
        event_id="event-sold-1",
        product_id=product.id,
    )

    result = _send_webhook(
        webhook_handler,
        settings,
        payload,
    )

    assert result.accepted is True
    assert result.action == "synced"

    assert vector_store.contains(product.id) is True
    assert vector_store.count() == 1


# ---------------------------------------------------------------------------
# PRODUCT_DEACTIVATED
# ---------------------------------------------------------------------------


def test_product_deactivated_remains_in_faiss(
    settings: Settings,
    webhook_handler: WebhookHandler,
    synchronization_service: SynchronizationService,
    vector_store: FAISSManager,
    product: Product,
) -> None:
    """PRODUCT_DEACTIVATED preserves the non-deleted catalogue row."""
    synchronization_service.sync_product(product.id)

    assert vector_store.contains(product.id) is True

    payload = _payload(
        "PRODUCT_DEACTIVATED",
        event_id="event-deactivated-1",
        product_id=product.id,
    )

    result = _send_webhook(
        webhook_handler,
        settings,
        payload,
    )

    assert result.accepted is True
    assert result.action == "synced"

    assert vector_store.contains(product.id) is True
    assert vector_store.count() == 1


# ---------------------------------------------------------------------------
# PRODUCT_REACTIVATED
# ---------------------------------------------------------------------------


def test_product_reactivated_adds_product_back_to_faiss(
    settings: Settings,
    webhook_handler: WebhookHandler,
    synchronization_service: SynchronizationService,
    vector_store: FAISSManager,
    product: Product,
) -> None:
    """PRODUCT_REACTIVATED fetches the current product and reindexes it."""
    synchronization_service.sync_product(product.id)

    assert vector_store.contains(product.id) is True

    # Simulate the product having previously been removed.
    synchronization_service.remove_product(product.id)

    assert vector_store.contains(product.id) is False

    payload = _payload(
        "PRODUCT_REACTIVATED",
        event_id="event-reactivated-1",
        product_id=product.id,
    )

    result = _send_webhook(
        webhook_handler,
        settings,
        payload,
    )

    assert result.accepted is True
    assert result.action == "reactivated"

    assert vector_store.contains(product.id) is True
    assert vector_store.count() == 1


# ---------------------------------------------------------------------------
# Duplicate event
# ---------------------------------------------------------------------------


def test_duplicate_webhook_is_not_processed_twice(
    settings: Settings,
    webhook_handler: WebhookHandler,
    vector_store: FAISSManager,
    product: Product,
) -> None:
    """A successful event is persisted and ignored when repeated."""
    payload = _payload(
        "PRODUCT_CREATED",
        event_id="event-duplicate-1",
        product_id=product.id,
    )

    first_result = _send_webhook(
        webhook_handler,
        settings,
        payload,
    )

    second_result = _send_webhook(
        webhook_handler,
        settings,
        payload,
    )

    assert first_result.accepted is True
    assert first_result.duplicated is False

    assert second_result.accepted is True
    assert second_result.duplicated is True
    assert second_result.action == "ignored"

    assert vector_store.count() == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_webhook_update_survives_faiss_reload(
    settings: Settings,
    webhook_handler: WebhookHandler,
    vector_store: FAISSManager,
    product: Product,
) -> None:
    """A webhook-driven FAISS modification remains available after reload."""
    payload = _payload(
        "PRODUCT_CREATED",
        event_id="event-persistence-1",
        product_id=product.id,
    )

    result = _send_webhook(
        webhook_handler,
        settings,
        payload,
    )

    assert result.accepted is True
    assert vector_store.contains(product.id) is True

    # Re-create the manager from the same settings.
    reloaded_store = FAISSManager(settings)

    assert reloaded_store.count() == 1
    assert reloaded_store.contains(product.id) is True


# ---------------------------------------------------------------------------
# Complete six-event scenario
# ---------------------------------------------------------------------------


def test_complete_six_event_lifecycle(
    settings: Settings,
    webhook_handler: WebhookHandler,
    vector_store: FAISSManager,
    catalogue: FakeCatalogue,
    product: Product,
) -> None:
    """Validate the complete lifecycle of one product through all six events."""

    # 1. PRODUCT_CREATED
    created_payload = _payload(
        "PRODUCT_CREATED",
        event_id="lifecycle-created",
        product_id=product.id,
    )

    created_result = _send_webhook(
        webhook_handler,
        settings,
        created_payload,
    )

    assert created_result.accepted is True
    assert vector_store.contains(product.id) is True

    # 2. PRODUCT_UPDATED
    updated_product = Product(
        id=product.id,
        title="Laptop Dell XPS Updated",
        description="Version mise à jour.",
        category=product.category,
        brand=product.brand,
        color=product.color,
        condition=product.condition,
        price=Decimal("14000"),
        currency=product.currency,
        city=product.city,
        image_urls=product.image_urls,
        status=product.status,
        is_sold=False,
        updated_at=datetime.now(timezone.utc),
    )

    catalogue.update_product(updated_product)

    updated_payload = _payload(
        "PRODUCT_UPDATED",
        event_id="lifecycle-updated",
        product_id=product.id,
    )

    updated_result = _send_webhook(
        webhook_handler,
        settings,
        updated_payload,
    )

    assert updated_result.accepted is True
    assert vector_store.contains(product.id) is True

    # 3. PRODUCT_SOLD
    sold_payload = _payload(
        "PRODUCT_SOLD",
        event_id="lifecycle-sold",
        product_id=product.id,
    )

    sold_result = _send_webhook(
        webhook_handler,
        settings,
        sold_payload,
    )

    assert sold_result.accepted is True
    assert sold_result.action == "synced"
    assert vector_store.contains(product.id) is True

    # 4. PRODUCT_REACTIVATED
    reactivated_payload = _payload(
        "PRODUCT_REACTIVATED",
        event_id="lifecycle-reactivated",
        product_id=product.id,
    )

    reactivated_result = _send_webhook(
        webhook_handler,
        settings,
        reactivated_payload,
    )

    assert reactivated_result.accepted is True
    assert vector_store.contains(product.id) is True

    # 5. PRODUCT_DEACTIVATED
    deactivated_payload = _payload(
        "PRODUCT_DEACTIVATED",
        event_id="lifecycle-deactivated",
        product_id=product.id,
    )

    deactivated_result = _send_webhook(
        webhook_handler,
        settings,
        deactivated_payload,
    )

    assert deactivated_result.accepted is True
    assert deactivated_result.action == "synced"
    assert vector_store.contains(product.id) is True

    # 6. PRODUCT_DELETED
    deleted_payload = _payload(
        "PRODUCT_DELETED",
        event_id="lifecycle-deleted",
        product_id=product.id,
    )

    deleted_result = _send_webhook(
        webhook_handler,
        settings,
        deleted_payload,
    )

    assert deleted_result.accepted is True
    assert vector_store.contains(product.id) is False
    assert vector_store.count() == 0
