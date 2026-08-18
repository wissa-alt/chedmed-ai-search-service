"""Unit tests for business-only ChedMed-to-vector synchronisation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import numpy as np
import pytest

from models.product import Product
from search.product_text_builder import ProductTextBuilder
from services.sync_service import SynchronizationService


@pytest.fixture
def product() -> Product:
    """Create a valid product without involving HTTP or embeddings infrastructure."""
    return Product(
        id="product-a",
        title="Vélo électrique",
        description="Très bon état.",
        category="Vélos",
        brand=None,
        color=None,
        condition=None,
        price=Decimal("1250"),
        currency="MAD",
        city=None,
        image_urls=(),
        status="ACTIVE",
        is_sold=False,
        updated_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
    )


@pytest.fixture
def api_client() -> Mock:
    """Return an API client double."""
    return Mock()


@pytest.fixture
def embedder() -> Mock:
    """Return an embedding service double with a stable vector."""
    fake_embedder = Mock()
    fake_embedder.embed_product.return_value = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )
    return fake_embedder


@pytest.fixture
def vector_store() -> Mock:
    """Return a FAISS manager double that defaults to an absent product."""
    fake_store = Mock()
    fake_store.contains.return_value = False
    fake_store.add.return_value = 7
    fake_store.update.return_value = 7
    return fake_store


@pytest.fixture
def text_builder() -> Mock:
    """Return a product text builder double."""
    fake_builder = Mock(spec=ProductTextBuilder)
    fake_builder.build.return_value = "Produit de test"
    return fake_builder


@pytest.fixture
def service(
    api_client: Mock,
    embedder: Mock,
    vector_store: Mock,
    text_builder: Mock,
) -> SynchronizationService:
    """Construct the service solely through dependency injection."""
    return SynchronizationService(
        api_client,
        embedder,
        vector_store,
        text_builder,
    )


def test_full_sync_indexes_all_products(
    service: SynchronizationService,
    api_client: Mock,
    embedder: Mock,
    vector_store: Mock,
    product: Product,
) -> None:
    """A full sync fetches products page by page, then saves once."""
    second_product = replace(product, id="product-b")

    api_client.get_products_page.side_effect = [
        [product, second_product],
    ]

    report = service.full_sync()

    assert report.total_products == 2
    assert report.indexed_products == 2
    assert report.updated_products == 0
    assert report.failed_products == 0

    api_client.get_products_page.assert_called_once_with(1, 500)

    vector_store.clear.assert_called_once()
    vector_store.save.assert_called_once()

    assert embedder.embed_product.call_count == 2


def test_full_sync_prefers_all_status_catalogue_page_by_default(
    embedder: Mock,
    vector_store: Mock,
    text_builder: Mock,
    product: Product,
) -> None:
    class AllStatusSource:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def get_products_page_any_status(self, page: int, limit: int) -> list[Product]:
            self.calls.append((page, limit))
            return [replace(product, status="PENDING", is_sold=True)]

        def get_products_page(self, page: int, limit: int) -> list[Product]:
            raise AssertionError("active-only page must not be used")

    source = AllStatusSource()
    report = SynchronizationService(
        source, embedder, vector_store, text_builder
    ).full_sync()

    assert report.total_products == 1
    assert report.indexed_products == 1
    assert source.calls == [(1, 500)]


def test_targeted_sync_reads_non_deleted_product_regardless_of_status(
    embedder: Mock,
    vector_store: Mock,
    text_builder: Mock,
    product: Product,
) -> None:
    class AllStatusSource:
        def get_product_any_status(self, product_id: str) -> Product:
            assert product_id == product.id
            return replace(product, status="REJECTED", is_sold=True)

        def get_product(self, product_id: str) -> Product:
            raise AssertionError("active-only lookup must not be used")

    result = SynchronizationService(
        AllStatusSource(), embedder, vector_store, text_builder
    ).sync_product(product.id)

    assert result.action == "added"
    vector_store.add.assert_called_once()


def test_full_sync_fetches_multiple_pages(
    service: SynchronizationService,
    api_client: Mock,
    vector_store: Mock,
    product: Product,
) -> None:
    """A full sync continues through multiple catalogue pages."""
    product_b = replace(product, id="product-b")
    product_c = replace(product, id="product-c")

    api_client.get_products_page.side_effect = [
        [product, product_b],
        [product_c],
    ]

    report = service.full_sync(page_size=2)

    assert report.total_products == 3
    assert report.indexed_products == 3
    assert report.updated_products == 0
    assert report.failed_products == 0

    assert api_client.get_products_page.call_args_list == [
        ((1, 2), {}),
        ((2, 2), {}),
    ]

    vector_store.clear.assert_called_once()
    vector_store.save.assert_called_once()


def test_sync_product_updates_existing_vector(
    service: SynchronizationService,
    api_client: Mock,
    vector_store: Mock,
    product: Product,
) -> None:
    """A targeted sync updates rather than duplicates an existing product."""
    api_client.get_product.return_value = product
    vector_store.contains.return_value = True
    vector_store.update.return_value = 12

    result = service.sync_product("product-a")

    assert result.product_id == "product-a"
    assert result.action == "updated"
    assert result.vector_id == 12

    api_client.get_product.assert_called_once_with("product-a")
    vector_store.update.assert_called_once()
    vector_store.add.assert_not_called()
    vector_store.save.assert_called_once()


def test_remove_product_removes_and_saves(
    service: SynchronizationService,
    vector_store: Mock,
) -> None:
    """Deletion of an indexed product is made durable immediately."""
    vector_store.contains.return_value = True

    result = service.remove_product("product-a")

    assert result.product_id == "product-a"
    assert result.action == "removed"

    vector_store.remove.assert_called_once_with("product-a")
    vector_store.save.assert_called_once()


def test_remove_product_skips_absent_product(
    service: SynchronizationService,
    vector_store: Mock,
) -> None:
    """Removing a product absent from FAISS does not modify the index."""
    vector_store.contains.return_value = False

    result = service.remove_product("product-a")

    assert result.product_id == "product-a"
    assert result.action == "skipped"

    vector_store.remove.assert_not_called()
    vector_store.save.assert_not_called()


def test_reactivate_product_delegates_to_targeted_sync(
    service: SynchronizationService,
    api_client: Mock,
    vector_store: Mock,
    product: Product,
) -> None:
    """Reactivation uses the current product source of truth and indexes it."""
    api_client.get_product.return_value = product
    vector_store.contains.return_value = False

    result = service.reactivate_product("product-a")

    assert result.product_id == "product-a"
    assert result.action == "added"

    api_client.get_product.assert_called_once_with("product-a")
    vector_store.add.assert_called_once()
    vector_store.save.assert_called_once()


def test_sync_updated_products_adds_and_updates_in_one_save(
    service: SynchronizationService,
    api_client: Mock,
    vector_store: Mock,
    product: Product,
) -> None:
    """Incremental sync processes every API result and persists at batch end."""
    second_product = replace(product, id="product-b")

    api_client.get_products_updated_after.return_value = [
        product,
        second_product,
    ]

    vector_store.contains.side_effect = [
        False,
        True,
    ]

    report = service.sync_updated_products(
        "2026-07-28T00:00:00+00:00"
    )

    assert report.total_products == 2
    assert report.indexed_products == 1
    assert report.updated_products == 1
    assert report.failed_products == 0

    assert vector_store.add.call_count == 1
    assert vector_store.update.call_count == 1

    vector_store.save.assert_called_once()


def test_full_sync_records_product_error_and_continues(
    service: SynchronizationService,
    api_client: Mock,
    embedder: Mock,
    vector_store: Mock,
    product: Product,
) -> None:
    """One failed product is reported while later products continue."""
    failing_product = replace(
        product,
        id="product-failing",
    )

    api_client.get_products_page.side_effect = [
        [failing_product, product],
    ]

    def embed_side_effect(current_product: Product) -> np.ndarray:
        if current_product.id == "product-failing":
            raise RuntimeError("embedding unavailable")

        return np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )

    embedder.embed_product.side_effect = embed_side_effect

    report = service.full_sync()

    assert report.total_products == 2
    assert report.failed_products == 1
    assert report.indexed_products == 1
    assert report.updated_products == 0

    assert len(report.errors) == 1
    assert report.errors[0].product_id == "product-failing"
    assert report.errors[0].message == "embedding unavailable"

    vector_store.add.assert_called_once()

    assert vector_store.add.call_args.args[0] == "product-a"

    assert np.array_equal(
        vector_store.add.call_args.args[1],
        np.array([1.0, 0.0], dtype=np.float32),
    )

    vector_store.save.assert_called_once()


def test_full_sync_rejects_invalid_page_size(
    service: SynchronizationService,
) -> None:
    """Full sync rejects non-positive page sizes."""
    with pytest.raises(ValueError, match="page_size"):
        service.full_sync(page_size=0)
