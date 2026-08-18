"""Tests for strict filtering on fields genuinely represented by Product."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from models.product import Product
from models.query_analysis import QueryIntent, SupportedLanguage
from models.search_query import SearchSource, StructuredSearchQuery
from search.filter_engine import FilterEngine


def _product(**overrides: object) -> Product:
    values = dict(
        id="1", title="Black Peshawari Chappal", description="Leather footwear",
        category="Hommes", brand="Adidas", color="Noir", condition="neuf",
        price=Decimal("110"), currency="MAD", city="Hrara", image_urls=(),
        status="ACTIVE", is_sold=False,
        updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    values.update(overrides)
    return Product(**values)  # type: ignore[arg-type]


def _query(**overrides: object) -> StructuredSearchQuery:
    query = StructuredSearchQuery(
        original_query="shoes for men", semantic_query="shoes for men",
        language=SupportedLanguage.ENGLISH, intent=QueryIntent.PRODUCT_SEARCH,
    )
    return replace(query, **overrides)


def test_product_type_is_never_a_strict_lexical_filter() -> None:
    """A semantic footwear candidate survives without any known alias."""
    matches, reason = FilterEngine().evaluate(
        _product(), _query(category="Hommes", product_type="chaussures")
    )
    assert matches
    assert reason is None


def test_resolved_category_remains_strict() -> None:
    matches, reason = FilterEngine().evaluate(
        _product(category="Femmes"), _query(category="Hommes", product_type="chaussures")
    )
    assert not matches
    assert reason == "category"


@pytest.mark.parametrize("catalogue_brand", [None, "Sans Marque", "Generic"])
def test_brand_visible_in_product_text_survives_unreliable_metadata(
    catalogue_brand: str | None,
) -> None:
    matches, reason = FilterEngine().evaluate(
        _product(
            title="Laptop Dell XPS",
            description="Portable Dell professionnel",
            category="Électronique",
            brand=catalogue_brand,
        ),
        _query(brand="Dell", category="Électronique"),
    )

    assert matches
    assert reason is None


def test_missing_brand_metadata_is_inconclusive_not_a_rejection() -> None:
    matches, reason = FilterEngine().evaluate(
        _product(brand=None),
        _query(brand="Nouvelle Marque"),
    )

    assert matches
    assert reason is None


def test_explicit_different_brand_is_a_reliable_conflict() -> None:
    matches, reason = FilterEngine().evaluate(
        _product(title="Adidas Superstar", brand="Adidas"),
        _query(brand="Nike"),
    )

    assert not matches
    assert reason == "brand"


def test_exact_catalogue_brand_match_is_accepted() -> None:
    matches, reason = FilterEngine().evaluate(
        _product(title="Running shoe", brand="Nike"),
        _query(brand="Nike"),
    )

    assert matches
    assert reason is None


def test_text_color_remains_a_strict_constraint() -> None:
    matches, reason = FilterEngine().evaluate(
        _product(color="Bleu"),
        _query(color="Rouge", source=SearchSource.TEXT),
    )

    assert not matches
    assert reason == "color"


def test_image_color_is_an_observation_not_an_exclusion_filter() -> None:
    matches, reason = FilterEngine().evaluate(
        _product(color="Noir"),
        _query(color="gray", source=SearchSource.IMAGE),
    )

    assert matches
    assert reason is None


@pytest.mark.parametrize(
    ("query_changes", "product_changes", "reason"),
    (
        ({"max_price": 100.0}, {}, "price"),
        ({"city": "Rabat"}, {}, "city"),
        ({"color": "Rouge"}, {}, "color"),
    ),
)
def test_reliable_product_fields_still_filter(
    query_changes: dict[str, object],
    product_changes: dict[str, object],
    reason: str,
) -> None:
    matches, rejected_field = FilterEngine().evaluate(
        _product(**product_changes), _query(**query_changes)
    )
    assert not matches
    assert rejected_field == reason
