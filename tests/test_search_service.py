"""Unit tests for the typed semantic-search orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import numpy as np
import pytest

from models.product import Product
from models.query_analysis import (
    QueryAnalysis,
    QueryIntent,
    SearchFilters,
    SupportedLanguage,
)
from models.search_query import SearchSource
from search.search_service import SearchService, SearchServiceError


@pytest.fixture
def product() -> Product:
    """Return one current product supplied by the catalogue source of truth."""
    return Product(
        id="product-1",
        title="Vélo électrique",
        description="Autonomie de 70 km.",
        category="Vélos",
        brand=None,
        color=None,
        condition=None,
        price=Decimal("1000"),
        currency="MAD",
        city=None,
        image_urls=(),
        status="ACTIVE",
        is_sold=False,
        updated_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def analysis() -> QueryAnalysis:
    """Return a real analysis rather than a permissive mock."""
    return QueryAnalysis(
        original_query="vélo électrique",
        language=SupportedLanguage.FRENCH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="vélo électrique",
        keywords=[],
        filters=SearchFilters(),
    )


@pytest.fixture
def dependencies() -> tuple[Mock, Mock, Mock, Mock, Mock]:
    """Return controlled infrastructure collaborators."""
    lookup = Mock()
    embedder = Mock()
    embedder.embed_query.return_value = np.array([1.0, 0.0], dtype=np.float32)
    vector_store = Mock()
    query_understanding = Mock()
    query_expansion = Mock()
    query_expansion.expand.side_effect = lambda query, **_: query
    return lookup, embedder, vector_store, query_understanding, query_expansion


def _service(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    analysis: QueryAnalysis,
    default_top_k: int = 5,
) -> SearchService:
    """Build one service with a real analysis at the understanding boundary."""
    lookup, embedder, vector_store, understanding, expansion = dependencies
    understanding.understand.return_value = analysis
    return SearchService(
        lookup,
        embedder,
        vector_store,
        understanding,
        expansion,
        default_top_k=default_top_k,
    )


def test_search_uses_real_query_analysis_and_preserves_scores(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    analysis: QueryAnalysis,
    product: Product,
) -> None:
    """The complete typed pipeline retrieves a current product by FAISS ID."""
    lookup, embedder, vector_store, understanding, expansion = dependencies
    vector_store.search.return_value = [("product-1", 0.91)]
    lookup.get_product.return_value = product

    result = _service(dependencies, analysis).search("  vélo électrique  ")

    assert result.query == "vélo électrique"
    assert result.items[0].product == product
    assert result.items[0].score == 0.91
    assert result.items[0].relevance_reason == "focal_lexical_evidence"
    understanding.understand.assert_called_once_with("vélo électrique")
    expansion.expand.assert_called_once_with(
        "vélo électrique",
        product_type=None,
        category=None,
        include_category_aliases=False,
    )
    embedder.embed_query.assert_called_once_with("vélo électrique")
    vector_store.search.assert_called_once_with(embedder.embed_query.return_value, 50)


def test_search_does_not_inject_normalized_category_into_embedding_query(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
) -> None:
    """Official filter labels never distort the natural-language embedding query."""
    _, embedder, vector_store, _, expansion = dependencies
    vector_store.search.return_value = []
    expansion.expand.return_value = "laptop gaming ordinateur portable"
    typed_analysis = QueryAnalysis(
        original_query="laptop gaming",
        language=SupportedLanguage.FRENCH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="laptop gaming",
        filters=SearchFilters(
            category_raw="laptop",
            category_normalized="Électronique",
        ),
    )

    _service(dependencies, typed_analysis).search("laptop gaming")

    embedded_query = embedder.embed_query.call_args.args[0]
    assert "Électronique" not in embedded_query
    assert "laptop gaming" in embedded_query


def test_search_does_not_embed_provider_hallucinations(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
) -> None:
    """Retrieval text is anchored to the original query, not an LLM rewrite."""
    _, embedder, vector_store, _, expansion = dependencies
    vector_store.search.return_value = []
    expansion.expand.side_effect = lambda query, **_: query
    typed_analysis = QueryAnalysis(
        original_query="bghit sberdila dial rjal",
        language=SupportedLanguage.DARIJA,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="bghit sberdila dial rjal ou casquettes pour hommes",
        keywords=["sberdila", "casquettes"],
        filters=SearchFilters(category_raw="rjal", product_type_raw="sberdila"),
    )

    _service(dependencies, typed_analysis).search("bghit sberdila dial rjal")

    embedded_query = embedder.embed_query.call_args.args[0]
    assert embedded_query == "sberdila"
    assert "casquettes" not in embedded_query


def test_search_rejects_unvalidated_llm_product_type_hint(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
) -> None:
    """An LLM cannot turn a brand-only request into footwear retrieval."""
    _, embedder, vector_store, _, expansion = dependencies
    vector_store.search.return_value = []
    expansion.expand.side_effect = lambda query, **_: query
    typed_analysis = QueryAnalysis(
        original_query="bghit adidas dial rjal",
        language=SupportedLanguage.DARIJA,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="chaussures de sport adidas homme",
        filters=SearchFilters(
            category_raw="rjal",
            product_type_raw="chaussures de sport",
            brand="adidas",
        ),
    )

    result = _service(dependencies, typed_analysis).search(
        "bghit adidas dial rjal"
    )

    assert result.structured_query is not None
    assert result.structured_query.product_type is None
    embedded_query = embedder.embed_query.call_args.args[0]
    assert "adidas" in embedded_query
    assert "chauss" not in embedded_query


def test_search_keeps_visible_brand_when_catalogue_brand_is_generic(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    """A text brand remains semantic while generic metadata stays soft."""
    query = "Dell laptop"
    lookup, embedder, vector_store, _, expansion = dependencies
    laptop = replace(
        product,
        id="123",
        title="Laptop Dell XPS - TEST E2E WEBHOOK",
        description="Laptop dell xps",
        category="Électronique",
        brand="Sans Marque",
    )
    noise = replace(
        product,
        id="noise",
        title="Generic electronic product",
        description="Unrelated item",
        category="Électronique",
        brand=None,
    )
    products = {laptop.id: laptop, noise.id: noise}
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [("123", 0.84), ("noise", 0.82)]
    expansion.expand.side_effect = lambda value, **_: value
    typed_analysis = QueryAnalysis(
        original_query=query,
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query=query,
        filters=SearchFilters(brand="Dell", product_type_raw="laptop"),
    )

    result = _service(dependencies, typed_analysis).search(query)

    assert result.structured_query is not None
    assert result.structured_query.brand == "Dell"
    assert "Dell" in embedder.embed_query.call_args.args[0]
    assert result.filtered_products_count == 2
    assert [item.product.id for item in result.items] == ["123"]


def test_image_source_keeps_visual_color_as_semantic_hint(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    """Vision color disagreement cannot remove an otherwise relevant laptop."""
    query = "Dark gray Dell laptop"
    lookup, embedder, vector_store, _, expansion = dependencies
    laptop = replace(
        product,
        id="123",
        title="Laptop Dell XPS - TEST E2E WEBHOOK",
        description="Laptop dell xps",
        category="Électronique",
        brand="Sans Marque",
        color="Noir",
    )
    noise = replace(
        product,
        id="noise",
        title="Generic electronic product",
        description="Unrelated item",
        category="Électronique",
        brand=None,
        color="gray",
    )
    products = {laptop.id: laptop, noise.id: noise}
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [("123", 0.84), ("noise", 0.82)]
    expansion.expand.side_effect = lambda value, **_: value
    typed_analysis = QueryAnalysis(
        original_query=query,
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query=query,
        filters=SearchFilters(
            brand="Dell", color="gray", product_type_raw="laptop"
        ),
    )

    result = _service(dependencies, typed_analysis).search(
        query, source=SearchSource.IMAGE
    )

    assert result.structured_query is not None
    assert result.structured_query.source is SearchSource.IMAGE
    assert result.structured_query.color == "gray"
    assert embedder.embed_query.call_args.args[0] == query
    assert result.filtered_products_count == 2
    assert [item.product.id for item in result.items] == ["123"]


def test_search_uses_explicit_limit(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    analysis: QueryAnalysis,
) -> None:
    """An explicit result limit overrides the configured default."""
    _, embedder, vector_store, _, _ = dependencies
    vector_store.search.return_value = []

    result = _service(dependencies, analysis).search("vélo", top_k=2)

    assert result.items == ()
    vector_store.search.assert_called_once_with(embedder.embed_query.return_value, 50)


def test_top_k_is_a_ceiling_not_a_fill_target(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    """topK=10 returns only the one candidate supported by relevance."""
    lookup, _, vector_store, _, _ = dependencies
    perfume = replace(
        product, id="120", title="Perfumes", description="Perfumes for everyone"
    )
    noise_one = replace(product, id="noise-1", title="Test Product")
    noise_two = replace(product, id="noise-2", title="Generic Item")
    products = {item.id: item for item in (perfume, noise_one, noise_two)}
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [
        ("120", 0.849), ("noise-1", 0.806), ("noise-2", 0.801)
    ]
    typed_analysis = QueryAnalysis(
        original_query="parfum",
        language=SupportedLanguage.FRENCH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="parfum",
        filters=SearchFilters(product_type_raw="parfum"),
    )

    result = _service(dependencies, typed_analysis).search("parfum", top_k=10)

    assert [item.product.id for item in result.items] == ["120"]
    assert result.relevant_products_count == 1
    assert len(result.items) == 1
    vector_store.search.assert_called_once_with(
        dependencies[1].embed_query.return_value, 100
    )


def test_similar_fallback_relaxes_color_but_preserves_product_evidence(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    """A colour mismatch may yield a related item, without admitting noise."""
    lookup, _, vector_store, _, _ = dependencies
    related = replace(
        product,
        id="related",
        title="Casual cotton shirt",
        description="Everyday shirt",
        category="Hommes",
        color="Bleu",
    )
    noise = replace(
        product,
        id="noise",
        title="Unrelated object",
        description="Generic catalogue item",
        category="Hommes",
        color="Rouge",
    )
    products = {item.id: item for item in (related, noise)}
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [("related", 0.84), ("noise", 0.81)]
    typed_analysis = QueryAnalysis(
        original_query="red shirt for men",
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="red shirt for men",
        filters=SearchFilters(
            category_raw="men", product_type_raw="shirt", color="Rouge"
        ),
    )

    result = _service(dependencies, typed_analysis).search("red shirt for men")

    assert result.primary_results_count == 0
    assert result.similar_results_count == 1
    assert result.match_type == "similar"
    assert [(item.product.id, item.match_type) for item in result.items] == [
        ("related", "similar")
    ]


def test_similar_fallback_keeps_explicit_price_authoritative(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    """A close product above an explicit ceiling cannot fill the response."""
    lookup, _, vector_store, _, _ = dependencies
    expensive = replace(
        product,
        id="expensive",
        title="Gaming computer",
        description="Powerful gaming computer",
        category="Électronique",
        price=Decimal("9000"),
    )
    lookup.get_product.return_value = expensive
    vector_store.search.return_value = [("expensive", 0.90)]
    typed_analysis = QueryAnalysis(
        original_query="gaming computer under 5000 MAD",
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="gaming computer under 5000 MAD",
        filters=SearchFilters(
            product_type_raw="computer",
            max_price=Decimal("5000"),
            currency="MAD",
        ),
    )

    result = _service(dependencies, typed_analysis).search(
        "gaming computer under 5000 MAD"
    )

    assert result.items == ()
    assert result.match_type == "none"


def test_marketplace_without_top_k_returns_all_useful_results(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    """The marketplace default is a ceiling of five, never an obligation."""
    lookup, _, vector_store, _, _ = dependencies
    products = {
        str(index): replace(
            product,
            id=str(index),
            title=f"Electric bicycle model {index}",
            description="Electric bicycle",
        )
        for index in range(8)
    }
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [
        (str(index), 0.90 - index * 0.001) for index in range(8)
    ]

    result = _service(dependencies, analysis=QueryAnalysis(
        original_query="electric bicycle",
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="electric bicycle",
        filters=SearchFilters(),
    )).search("electric bicycle")

    assert len(result.items) == 8
    assert all(item.match_type == "exact" for item in result.items)


@pytest.mark.parametrize("query", ["t-shirt", "sweater"])
def test_family_fallback_returns_only_related_tops(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
    query: str,
) -> None:
    lookup, _, vector_store, _, _ = dependencies
    hoodie = replace(product, id="hoodie", title="Casual hoodie", category="Hommes")
    related_title = "Plain sweater" if query == "t-shirt" else "Plain sweatshirt"
    related = replace(
        product, id="related", title=related_title, category="Hommes"
    )
    laptop = replace(
        product, id="laptop", title="Gaming laptop", category="Électronique"
    )
    products = {item.id: item for item in (hoodie, related, laptop)}
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [
        ("laptop", 0.84), ("hoodie", 0.835), ("related", 0.832)
    ]
    typed_analysis = QueryAnalysis(
        original_query=query,
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query=query,
        filters=SearchFilters(product_type_raw=query),
    )

    result = _service(dependencies, typed_analysis).search(query)

    assert result.primary_results_count == 0
    assert result.match_type == "similar"
    assert [item.product.id for item in result.items] == ["hoodie", "related"]
    assert all(item.relevance_reason == "product_family:tops" for item in result.items)


def test_family_fallback_maps_resolved_footwear_to_catalogue_chappal(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    lookup, _, vector_store, _, _ = dependencies
    hoodie = replace(product, id="hoodie", title="Men hoodie", category="Hommes")
    chappal = replace(
        product, id="chappal", title="Black Peshawari Chappal", category="Hommes"
    )
    products = {item.id: item for item in (hoodie, chappal)}
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [("hoodie", 0.84), ("chappal", 0.839)]
    typed_analysis = QueryAnalysis(
        original_query="chaussures pour hommes",
        language=SupportedLanguage.FRENCH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="chaussures pour hommes",
        filters=SearchFilters(category_raw="hommes", product_type_raw="chaussures"),
    )

    result = _service(dependencies, typed_analysis).search("chaussures pour hommes")

    assert [item.product.id for item in result.items] == ["chappal"]
    assert result.items[0].match_type == "similar"
    assert result.items[0].relevance_reason == "product_family:footwear"


def test_family_fallback_never_crosses_reliable_audience_filter(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    lookup, _, vector_store, _, _ = dependencies
    mens_perfume = replace(
        product, id="perfume", title="Perfumes", category="Hommes"
    )
    womens_bag = replace(product, id="bag", title="Women bag", category="Femmes")
    products = {item.id: item for item in (mens_perfume, womens_bag)}
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [("perfume", 0.85), ("bag", 0.84)]
    typed_analysis = QueryAnalysis(
        original_query="parfum pour femme",
        language=SupportedLanguage.FRENCH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="parfum pour femme",
        filters=SearchFilters(category_raw="femme", product_type_raw="parfum"),
    )

    result = _service(dependencies, typed_analysis).search("parfum pour femme")

    assert result.items == ()
    assert result.match_type == "none"


def test_family_fallback_without_top_k_returns_all_related_products(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    lookup, _, vector_store, _, _ = dependencies
    products = {
        str(index): replace(product, id=str(index), title=f"Casual hoodie {index}")
        for index in range(7)
    }
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [
        (str(index), 0.84 - index * 0.001) for index in range(7)
    ]
    typed_analysis = QueryAnalysis(
        original_query="sweater",
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="sweater",
        filters=SearchFilters(product_type_raw="sweater"),
    )

    result = _service(dependencies, typed_analysis).search("sweater")

    assert len(result.items) == 7
    assert result.similar_results_count == 7
    assert all(item.match_type == "similar" for item in result.items)


def test_family_fallback_completes_a_partial_primary_without_crossing_family(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    """A strong exact result stays first and only compatible family items complete it."""
    lookup, _, vector_store, _, _ = dependencies
    shirt = replace(product, id="shirt", title="Classic t-shirt")
    hoodie = replace(product, id="hoodie", title="Casual hoodie")
    laptop = replace(product, id="laptop", title="Gaming laptop", category="Électronique")
    products = {item.id: item for item in (shirt, hoodie, laptop)}
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [
        ("shirt", 0.90), ("laptop", 0.86), ("hoodie", 0.84)
    ]
    typed_analysis = QueryAnalysis(
        original_query="t-shirt",
        language=SupportedLanguage.FRENCH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="t-shirt",
        filters=SearchFilters(product_type_raw="t-shirt"),
    )

    result = _service(dependencies, typed_analysis).search("t-shirt", top_k=5)

    assert [(item.product.id, item.match_type) for item in result.items] == [
        ("shirt", "exact"),
        ("hoodie", "similar"),
    ]
    assert result.primary_results_count == 1
    assert result.similar_results_count == 1
    assert result.match_type == "relevant"


def test_broad_fallback_is_bounded_to_controlled_product_domain(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    lookup, _, vector_store, _, _ = dependencies
    laptop_bag = replace(
        product, id="bag", title="Protective computer bag", category="Électronique"
    )
    perfume = replace(
        product, id="perfume", title="Luxury perfume", category="Électronique"
    )
    products = {item.id: item for item in (laptop_bag, perfume)}
    lookup.get_product.side_effect = products.__getitem__
    vector_store.search.return_value = [("perfume", 0.821), ("bag", 0.82)]
    typed_analysis = QueryAnalysis(
        original_query="ultrabook",
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="ultrabook",
        filters=SearchFilters(product_type_raw="ultrabook"),
    )

    result = _service(dependencies, typed_analysis).search("ultrabook")

    assert [(item.product.id, item.match_type) for item in result.items] == [
        ("bag", "broad_similar")
    ]
    assert result.match_type == "broad_similar"
    assert result.broad_similar_results_count == 1


def test_fallback_relaxes_only_an_inferred_category_for_related_accessory(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    lookup, _, vector_store, _, _ = dependencies
    laptop = replace(
        product, id="laptop", title="Dell laptop", category="Électronique",
        brand="Dell",
    )
    bag = replace(
        product, id="bag", title="Laptop messenger bag", category="Femmes",
        brand="Adidas",
    )
    lookup.get_product.side_effect = {"laptop": laptop, "bag": bag}.__getitem__
    vector_store.search.return_value = [("laptop", 0.90), ("bag", 0.84)]
    typed_analysis = QueryAnalysis(
        original_query="Dell laptop", language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH, expanded_query="Dell laptop",
        filters=SearchFilters(brand="Dell", product_type_raw="laptop", category_raw=None),
    )

    result = _service(dependencies, typed_analysis).search("Dell laptop")

    assert [(item.product.id, item.match_type) for item in result.items] == [
        ("laptop", "exact"), ("bag", "broad_similar")
    ]
    assert result.match_type == "relevant"
    assert result.broad_similar_results_count == 1


def test_search_normalizer_preserves_original_and_exposes_normalized_query(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    lookup, _, vector_store, understanding, _ = dependencies
    lookup.get_product.return_value = replace(product, title="Gaming laptop")
    vector_store.search.return_value = [("product-1", 0.9)]
    understanding.understand.return_value = QueryAnalysis(
        original_query="gaming laptop",
        language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="gaming laptop",
        filters=SearchFilters(),
    )
    normalizer = Mock()
    normalizer.normalize.return_value = "gaming laptop"
    service = _service(dependencies, understanding.understand.return_value)
    service._query_normalizer = normalizer

    result = service.search("geming labtop")

    normalizer.normalize.assert_called_once_with("geming labtop")
    understanding.understand.assert_called_once_with("gaming laptop")
    assert result.original_query == "geming labtop"
    assert result.normalized_query == "gaming laptop"


def test_include_all_returns_every_indexed_product_and_preserves_status(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    lookup, _, vector_store, _, _ = dependencies
    exact = replace(product, id="exact", title="Dell laptop", status="ACCEPTED")
    related = replace(product, id="related", title="HP laptop", status="PENDING")
    sold = replace(product, id="sold", title="Computer bag", status="SOLD", is_sold=True)
    unrelated = replace(product, id="other", title="Luxury perfume", status="PENDING")
    products = {item.id: item for item in (exact, related, sold, unrelated)}
    lookup.get_product_any_status.side_effect = products.__getitem__
    vector_store.count.return_value = 4
    vector_store.search.return_value = [
        ("other", 0.91), ("exact", 0.90), ("related", 0.85), ("sold", 0.80)
    ]
    typed_analysis = QueryAnalysis(
        original_query="Dell laptop", language=SupportedLanguage.ENGLISH,
        intent=QueryIntent.PRODUCT_SEARCH, expanded_query="Dell laptop",
        filters=SearchFilters(brand="Dell", product_type_raw="laptop"),
    )

    result = _service(dependencies, typed_analysis).search(
        "Dell laptop", include_all=True
    )

    assert len(result.items) == 4
    assert len({item.product.id for item in result.items}) == 4
    assert result.items[0].product.id == "exact"
    assert result.items[-1].match_type == "unrelated"
    assert {item.product.status for item in result.items} == {"ACCEPTED", "PENDING", "SOLD"}
    assert result.total_catalog_products == 4
    assert result.candidate_products_count == 4
    vector_store.search.assert_called_once_with(
        dependencies[1].embed_query.return_value, 4
    )


def test_normal_search_keeps_relevant_sold_and_pending_but_excludes_rejected(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    analysis: QueryAnalysis,
    product: Product,
) -> None:
    lookup, _, vector_store, _, _ = dependencies
    lookup.get_product.side_effect = {
        "active": product,
        "sold": replace(product, id="sold", is_sold=True, status="SOLD"),
        "pending": replace(product, id="pending", status="PENDING"),
        "rejected": replace(product, id="rejected", status="REJECTED"),
    }.__getitem__
    vector_store.count.return_value = 4
    vector_store.search.return_value = [
        ("sold", 0.94), ("pending", 0.93), ("rejected", 0.92), ("active", 0.91)
    ]

    result = _service(dependencies, analysis).search("vélo électrique")

    assert [item.product.id for item in result.items] == ["sold", "pending", "product-1"]
    assert result.items[0].product.is_sold is True


def test_search_keeps_results_when_category_is_unknown(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    product: Product,
) -> None:
    """An unresolved raw category skips only the category filter."""
    lookup, _, vector_store, _, _ = dependencies
    vector_store.search.return_value = [("product-1", 0.91)]
    lookup.get_product.return_value = replace(product, title="Astronautes")
    typed_analysis = QueryAnalysis(
        original_query="astronautes",
        language=SupportedLanguage.FRENCH,
        intent=QueryIntent.PRODUCT_SEARCH,
        expanded_query="astronautes",
        filters=SearchFilters(category_raw="astronautes"),
    )

    result = _service(dependencies, typed_analysis).search("astronautes")

    assert [item.product.id for item in result.items] == ["product-1"]


def test_search_skips_products_missing_from_source_of_truth(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    analysis: QueryAnalysis,
) -> None:
    """A stale vector ID cannot produce a stale public search result."""
    lookup, _, vector_store, _, _ = dependencies
    vector_store.search.return_value = [("missing-product", 0.7)]
    lookup.get_product.side_effect = RuntimeError("not found")

    result = _service(dependencies, analysis).search("vélo")

    assert result.items == ()
    lookup.get_product.assert_called_once_with("missing-product")


@pytest.mark.parametrize("query", ["", "   ", 1])
def test_search_rejects_invalid_query(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    analysis: QueryAnalysis,
    query: object,
) -> None:
    """Only non-empty string queries reach the typed pipeline."""
    with pytest.raises(ValueError):
        _service(dependencies, analysis).search(query)  # type: ignore[arg-type]


def test_search_translates_embedding_failures(
    dependencies: tuple[Mock, Mock, Mock, Mock, Mock],
    analysis: QueryAnalysis,
) -> None:
    """Infrastructure failures remain behind the SearchService boundary."""
    _, embedder, _, _, _ = dependencies
    embedder.embed_query.side_effect = RuntimeError("model unavailable")

    with pytest.raises(SearchServiceError):
        _service(dependencies, analysis).search("vélo")
