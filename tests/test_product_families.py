"""Tests for the intentionally small marketplace family taxonomy."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from models.product import Product
from models.query_analysis import QueryIntent, SupportedLanguage
from models.search_query import StructuredSearchQuery
from search.product_families import ProductFamilies


@pytest.fixture
def product() -> Product:
    return Product(
        id="1", title="Generic", description="Generic", category="Hommes",
        brand=None, color=None, condition=None, price=Decimal("100"),
        currency="MAD", city=None, image_urls=(), status="ACTIVE",
        is_sold=False, updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    ("query", "product_type", "expected"),
    [
        ("I want a t-shirt", None, "tops"),
        ("sweater", None, "tops"),
        ("baskets pour hommes", "chaussures", "footwear"),
        ("unknown new product", None, None),
    ],
)
def test_query_family_resolution(
    query: str, product_type: str | None, expected: str | None
) -> None:
    structured = StructuredSearchQuery(
        original_query=query,
        semantic_query=query,
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        product_type=product_type,
    )
    assert ProductFamilies().for_query(structured) == expected


def test_product_family_uses_catalogue_text(product: Product) -> None:
    families = ProductFamilies()
    assert families.for_product(replace(product, title="Casual hoodie")) == "tops"
    assert families.for_product(replace(product, title="Peshawari Chappal")) == "footwear"
    assert families.for_product(replace(product, title="Dell XPS laptop")) == "computers"
    assert families.for_product(replace(product, title="Perfumes")) == "fragrance"
    assert families.for_product(replace(product, title="Basket ball")) is None
