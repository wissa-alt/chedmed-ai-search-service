"""Tests for the canonical category ownership of query domain models."""

from __future__ import annotations

import pytest

from models.query_analysis import (
    QueryAnalysis,
    QueryIntent,
    SearchFilters,
    SupportedLanguage,
)


def _analysis(filters: SearchFilters) -> QueryAnalysis:
    """Create a minimal valid typed analysis."""
    return QueryAnalysis(
        original_query="laptop",
        language=SupportedLanguage.FRENCH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="laptop gaming",
        filters=filters,
    )


def test_categories_are_canonical_filter_fields_with_compatibility_views() -> None:
    """Raw and normalized categories remain distinct and serializable."""
    analysis = _analysis(
        SearchFilters(
            category_raw="laptops",
            category_normalized="Électronique",
        )
    )

    assert analysis.filters.category_raw == "laptops"
    assert analysis.filters.category_normalized == "Électronique"
    assert analysis.category_raw == "laptops"
    assert analysis.category_normalized == "Électronique"
    assert analysis.to_dict()["filters"]["category_raw"] == "laptops"
    assert analysis.to_dict()["filters"]["category_normalized"] == "Électronique"


def test_legacy_category_arguments_are_forwarded_to_filters() -> None:
    """Existing callers can retain the old QueryAnalysis constructor keywords."""
    analysis = QueryAnalysis(
        original_query="smartphone",
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="smartphone",
        category_raw="smartphone",
        category_normalized="Électronique",
    )

    assert analysis.filters.category_raw == "smartphone"
    assert analysis.filters.category_normalized == "Électronique"


def test_conflicting_legacy_and_filter_category_values_are_rejected() -> None:
    """Two sources cannot silently create inconsistent category state."""
    with pytest.raises(ValueError, match="category_raw"):
        QueryAnalysis(
            original_query="laptop",
            language=SupportedLanguage.FRENCH,
            intent=QueryIntent.PRODUCT_SEARCH,
            expanded_query="laptop",
            category_raw="laptop",
            filters=SearchFilters(category_raw="ordinateur portable"),
        )
