"""Unit tests for the PostgreSQL ``public.products`` repository."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from database.repositories.product_repository import ProductNotFoundError, ProductRepository


@pytest.fixture
def database() -> Mock:
    """Return an injected PostgreSQL boundary and its cursor double."""
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=None)
    database_boundary = Mock()
    database_boundary.cursor.return_value = cursor
    return database_boundary


def _row() -> dict[str, object]:
    """Return one row matching the repository SELECT aliases."""
    return {
        "id": "product-1", "title": "Titre français", "description": "Description française",
        "category": "category-1", "price": Decimal("100"), "discounted_price": Decimal("80"),
        "status": "ACTIVE", "is_sold": False, "updated_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
        "thumbnail_url": "https://images.test/product.jpg", "title_fr": "Titre français",
        "title_en": "English title", "title_ar": None, "description_fr": "Description française",
        "description_en": "English description", "description_ar": None,
    }


def test_get_all_products_maps_multilingual_columns(database: Mock) -> None:
    """Rows become typed products without exposing SQL outside the repository."""
    cursor = database.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [_row()]

    product = ProductRepository(database).get_all_products()[0]

    assert product.title == "Titre français"
    assert "English title" in product.description
    assert product.image_urls == ("https://images.test/product.jpg",)
    query = cursor.execute.call_args.args[0]
    assert "FROM public.products" in query
    assert "deleted_at IS NULL" in query
    assert "is_sold IS FALSE" in query


def test_get_product_only_reads_indexable_product(database: Mock) -> None:
    """The fixed SQL binds the identifier and retains active-product filters."""
    cursor = database.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [_row()]

    assert ProductRepository(database).get_product("product-1").id == "product-1"
    query, params = cursor.execute.call_args.args
    assert "id::text = %s" in query
    assert params == ("product-1",)


def test_missing_deleted_sold_or_inactive_product_is_not_found(database: Mock) -> None:
    """A filtered row is not allowed to re-enter the FAISS index."""
    cursor = database.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []

    with pytest.raises(ProductNotFoundError):
        ProductRepository(database).get_product("product-removed")


def test_incremental_sync_binds_updated_at(database: Mock) -> None:
    """Backup synchronisation uses a parameterised timestamp filter."""
    cursor = database.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [_row()]

    ProductRepository(database).get_products_updated_after("2026-06-01T00:00:00+00:00")

    query, params = cursor.execute.call_args.args
    assert "updated_at > %s" in query
    assert params == ("2026-06-01T00:00:00+00:00",)
