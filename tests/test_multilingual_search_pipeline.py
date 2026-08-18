"""Regression and integration coverage for multilingual product search."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import numpy as np
import pytest

from models.product import Product
from models.query_analysis import QueryAnalysis, QueryIntent, SearchFilters, SupportedLanguage
from search.category_resolver import CategoryResolver
from search.product_type_resolver import ProductTypeResolver
from search.query_expansion import QueryExpansionService
from search.search_service import SearchService


CASES = (
    ("bghit sberdila dial rjal", SupportedLanguage.DARIJA),
    ("bghit sberdila dyal rjal", SupportedLanguage.DARIJA),
    ("je cherche des chaussures pour homme", SupportedLanguage.FRENCH),
    ("chaussures homme", SupportedLanguage.FRENCH),
    ("men shoes", SupportedLanguage.ENGLISH),
    ("بغيت سبرديلة ديال الرجال", SupportedLanguage.DARIJA),
)


def _men_shoes() -> Product:
    return Product(
        id="men-shoes-1",
        title="Black Peshawari Chappal with Vibram Sole – 09211",
        description="Traditional leather footwear.",
        category="Hommes",
        brand=None,
        color=None,
        condition="neuf",
        price=Decimal("250"),
        currency="MAD",
        city="Casablanca",
        image_urls=(),
        status="ACTIVE",
        is_sold=False,
        updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def _noise_product(identifier: str = "noise-1") -> Product:
    product = _men_shoes()
    return Product(
        id=identifier,
        title="Unrelated catalogue item",
        description="Generic description.",
        category=product.category,
        brand=product.brand,
        color=product.color,
        condition=product.condition,
        price=product.price,
        currency=product.currency,
        city=product.city,
        image_urls=product.image_urls,
        status=product.status,
        is_sold=product.is_sold,
        updated_at=product.updated_at,
    )


@pytest.mark.parametrize(("query", "language"), CASES)
def test_multilingual_queries_resolve_consistently(query: str, language: SupportedLanguage) -> None:
    """All required spellings converge without relying on an LLM mapping."""
    assert CategoryResolver().resolve_query(query) == "Hommes"
    assert ProductTypeResolver().resolve_query(query) == "chaussures"
    assert language is not SupportedLanguage.UNKNOWN


def test_unresolved_hints_are_not_filters() -> None:
    """Unknown values remain hints and cannot become exclusion filters."""
    assert CategoryResolver().resolve("astronautes") is None
    assert ProductTypeResolver().resolve("objet-inconnu") is None


@pytest.mark.parametrize(("query", "language"), CASES)
def test_complete_mocked_pipeline_returns_matching_product(
    query: str,
    language: SupportedLanguage,
) -> None:
    """LLM -> expansion -> embedding -> FAISS -> API -> filter -> ranking."""
    understanding = Mock()
    understanding.understand.return_value = QueryAnalysis(
        original_query=query,
        language=language,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query=query,
        # Deliberately emulate imperfect/missing provider hints. Resolvers
        # must still recover from the original multilingual query.
        filters=SearchFilters(category_raw=None, product_type_raw=None),
        confidence=0.92,
    )
    embedder = Mock()
    embedder.embed_query.return_value = np.array([1.0, 0.0], dtype=np.float32)
    vector_store = Mock()
    vector_store.search.return_value = [
        ("men-shoes-1", 0.88),
        ("noise-1", 0.80),
    ]
    lookup = Mock()
    products = {"men-shoes-1": _men_shoes(), "noise-1": _noise_product()}
    lookup.get_product.side_effect = products.__getitem__
    expansion = QueryExpansionService()
    service = SearchService(lookup, embedder, vector_store, understanding, expansion, 5)

    result = service.search(query)

    assert result.structured_query is not None
    assert result.structured_query.language == language
    assert result.structured_query.category == "Hommes"
    assert result.structured_query.product_type == "chaussures"
    assert "chaussure" in result.structured_query.semantic_query.lower() or "shoes" in result.structured_query.semantic_query.lower()
    assert result.faiss_candidates_count == 2
    assert result.filtered_products_count == 2
    assert result.relevant_products_count == 1
    assert len(result.items) == 1
    assert result.items[0].product.category == "Hommes"


def test_regression_rjal_never_excludes_hommes_product() -> None:
    """The production bug cannot regress even with raw Darija LLM output."""
    understanding = Mock()
    understanding.understand.return_value = QueryAnalysis(
        original_query="bghit sberdila dial rjal",
        language=SupportedLanguage.DARIJA,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="bghit sberdila dial rjal",
        filters=SearchFilters(category_raw="rjal", product_type_raw="sberdila"),
    )
    embedder = Mock(embed_query=Mock(return_value=np.array([1.0], dtype=np.float32)))
    vector_store = Mock(search=Mock(return_value=[
        ("men-shoes-1", 0.9), ("noise-1", 0.8)
    ]))
    products = {"men-shoes-1": _men_shoes(), "noise-1": _noise_product()}
    lookup = Mock(get_product=Mock(side_effect=products.__getitem__))

    result = SearchService(
        lookup, embedder, vector_store, understanding, QueryExpansionService(), 5
    ).search("bghit sberdila dial rjal")

    assert result.structured_query.category == "Hommes"
    assert result.structured_query.product_type == "chaussures"
    assert len(result.items) > 0


@pytest.mark.parametrize(
    ("query", "title", "category"),
    (
        ("peshawari chappal homme", "Black Peshawari Chappal", "Hommes"),
        ("gaming laptop", "Laptop Dell XPS Gaming", "Électronique"),
        ("smartphone", "Flagship mobile device", "Électronique"),
    ),
)
def test_unknown_product_type_still_uses_semantic_retrieval(
    query: str,
    title: str,
    category: str,
) -> None:
    """Concepts absent from product_types.json never become exclusions."""
    assert ProductTypeResolver().resolve_query(query) is None
    product = _men_shoes()
    product = Product(
        id=product.id,
        title=title,
        description=product.description,
        category=category,
        brand=product.brand,
        color=product.color,
        condition=product.condition,
        price=product.price,
        currency=product.currency,
        city=product.city,
        image_urls=product.image_urls,
        status=product.status,
        is_sold=product.is_sold,
        updated_at=product.updated_at,
    )
    understanding = Mock()
    understanding.understand.return_value = QueryAnalysis(
        original_query=query,
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query=query,
        filters=SearchFilters(product_type_raw=query.split()[0]),
    )
    embedder = Mock(embed_query=Mock(return_value=np.array([1.0], dtype=np.float32)))
    vector_store = Mock(search=Mock(return_value=[
        (product.id, 0.87), ("noise-1", 0.78)
    ]))
    noise = _noise_product()
    if category != noise.category:
        noise = Product(
            id=noise.id, title=noise.title, description=noise.description,
            category=category, brand=noise.brand, color=noise.color,
            condition=noise.condition, price=noise.price, currency=noise.currency,
            city=noise.city, image_urls=noise.image_urls, status=noise.status,
            is_sold=noise.is_sold, updated_at=noise.updated_at,
        )
    products = {product.id: product, noise.id: noise}
    lookup = Mock(get_product=Mock(side_effect=products.__getitem__))

    result = SearchService(
        lookup, embedder, vector_store, understanding, QueryExpansionService(), 5
    ).search(query)

    assert result.structured_query.product_type is None
    assert [item.product.id for item in result.items] == [product.id]
