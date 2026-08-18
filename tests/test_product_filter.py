"""Tests for category filtering through normalized category state only."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from models.product import Product
from models.query_analysis import SearchFilters
from search.product_filter import ProductFilterService


def _product(category: str) -> Product:
    """Build a product with only the category varying between cases."""
    return Product(
        id=category,
        title="Produit",
        description="Description",
        category=category,
        brand=None,
        color=None,
        condition=None,
        price=Decimal("100"),
        currency="MAD",
        city=None,
        image_urls=(),
        status="ACTIVE",
        is_sold=False,
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_normalized_category_filters_products_strictly() -> None:
    """Only the official normalized category controls category acceptance."""
    filters = SearchFilters(
        category_raw="laptop",
        category_normalized="Électronique",
    )
    service = ProductFilterService()

    assert service.matches(_product("Électronique"), filters)
    assert not service.matches(_product("Maison"), filters)


def test_unknown_raw_category_skips_category_filter() -> None:
    """An unresolved category retains recall instead of excluding all products."""
    filters = SearchFilters(
        category_raw="astronautes",
        category_normalized=None,
    )

    assert ProductFilterService().matches(_product("Maison"), filters)
