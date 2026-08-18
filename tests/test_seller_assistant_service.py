"""Seller assistant pricing tests using the existing SearchService contract."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, call

import pytest

from app.services.seller_assistant_service import (
    SellerAssistantService,
    SellerAssistantUnavailableError,
    SellerAssistantValidationError,
    SellerProductInput,
)
from models.product import Product
from models.query_analysis import QueryIntent, SupportedLanguage
from models.search_query import SearchSource, StructuredSearchQuery
from search.search_service import SearchResult, SearchResultItem


@pytest.fixture
def product() -> Product:
    return Product(
        id="p", title="Marketplace item", description="Comparable item",
        category="Électronique", brand=None, color=None, condition="Bon état",
        price=Decimal("400"), currency="MAD", city=None, image_urls=(),
        status="ACCEPTED", is_sold=False,
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _seller(price: str = "420", **overrides: object) -> SellerProductInput:
    values = {
        "title": "Generic marketplace item",
        "description": "Produit en bon état",
        "category": None,
        "brand": None,
        "color": None,
        "condition": "Bon état",
        "seller_price": Decimal(price),
        "currency": "MAD",
        "product_id": None,
    }
    values.update(overrides)
    return SellerProductInput(**values)  # type: ignore[arg-type]


def _result(
    product: Product,
    prices: list[object],
    *,
    match_type: str = "relevant",
    titles: list[str] | None = None,
) -> SearchResult:
    items = tuple(
        SearchResultItem(
            replace(
                product,
                id=str(index),
                price=price,  # type: ignore[arg-type]
                title=titles[index] if titles else product.title,
            ),
            0.90 - index * 0.01,
            relevance_reason="focal_lexical_evidence",
            lexical_terms=("specific",),
            match_type=match_type,
        )
        for index, price in enumerate(prices)
    )
    return SearchResult("query", items, match_type=match_type)


def _assist(
    product: Product,
    prices: list[object],
    seller: SellerProductInput | None = None,
    *,
    match_type: str = "relevant",
    titles: list[str] | None = None,
    description_provider: object | None = None,
):
    search = Mock()
    search.search.return_value = _result(
        product, prices, match_type=match_type, titles=titles
    )
    service = SellerAssistantService(search, description_provider)  # type: ignore[arg-type]
    return service.assist(seller or _seller()), search


def test_median_and_quartile_range_are_based_on_catalogue_prices(product: Product) -> None:
    result, search = _assist(product, [350, 400, 420, 450, 500])
    assert result.estimated_price == Decimal("420")
    assert result.recommended_range is not None
    assert result.recommended_range.minimum == Decimal("400")
    assert result.recommended_range.maximum == Decimal("450")
    assert result.price_assessment == "reasonable"
    assert result.confidence == "good"
    search.search.assert_called_once()
    assert search.search.call_args.args[1] == 20


def test_iqr_removes_large_outlier(product: Product) -> None:
    result, _ = _assist(product, [350, 400, 420, 450, 500, 5000])
    assert result.estimated_price == Decimal("420")
    assert result.comparables_count == 5
    assert all(item.product.price != Decimal("5000") for item in result.comparables)


@pytest.mark.parametrize(
    ("seller_price", "expected"),
    [("420", "reasonable"), ("300", "too_low"), ("600", "too_high")],
)
def test_price_assessment_uses_observed_range(
    product: Product, seller_price: str, expected: str
) -> None:
    result, _ = _assist(product, [350, 400, 420, 450, 500], _seller(seller_price))
    assert result.price_assessment == expected


def test_zero_and_one_comparable_are_honest(product: Product) -> None:
    empty, _ = _assist(product, [])
    one, _ = _assist(product, [400])
    assert empty.estimated_price is None
    assert empty.price_assessment == "insufficient_data"
    assert empty.confidence == "none"
    assert one.estimated_price is None
    assert one.price_assessment == "insufficient_data"
    assert one.confidence == "very_low"
    assert one.comparables_count == 1


def test_family_only_comparables_reduce_confidence(product: Product) -> None:
    seller = _seller(title="Casual sweater", condition=None)
    result, _ = _assist(
        product,
        [350, 400, 420, 450, 500],
        seller,
        match_type="similar",
        titles=["Hoodie"] * 5,
    )
    assert result.estimated_price == Decimal("420")
    assert result.confidence == "low"


def test_invalid_prices_currency_and_explicit_brand_conflict_are_excluded(
    product: Product,
) -> None:
    valid = replace(product, id="valid", title="Nike shoes", brand="Nike")
    wrong_currency = replace(valid, id="eur", price=Decimal("300"), currency="EUR")
    zero = replace(valid, id="zero", price=Decimal("0"))
    negative = replace(valid, id="negative", price=Decimal("-1"))
    missing = replace(valid, id="missing", price=None)  # type: ignore[arg-type]
    conflict = replace(valid, id="adidas", title="Adidas shoes", brand="Adidas")
    search = Mock()
    search.search.return_value = SearchResult(
        "Nike shoes",
        tuple(
            SearchResultItem(
                item, 0.9,
                relevance_reason="focal_lexical_evidence",
                lexical_terms=("shoes",),
                match_type="relevant",
            )
            for item in (valid, wrong_currency, zero, negative, missing, conflict)
        ),
    )
    seller = _seller(title="Nike shoes", brand="Nike", condition=None)

    result = SellerAssistantService(search).assist(seller)

    assert result.comparables_count == 1
    assert result.comparables[0].product.id == "valid"


def test_description_failure_falls_back_without_breaking_pricing(product: Product) -> None:
    provider = Mock()
    provider.suggest_seller_description.side_effect = RuntimeError("provider down")
    seller = _seller(
        title="Dell XPS", description="laptop dell bon état",
        brand="Dell", color="Noir", condition="Très bon état",
    )
    laptop = replace(product, title="Dell XPS laptop", brand="Sans Marque")
    result, _ = _assist(laptop, [400, 500], seller, description_provider=provider)
    assert result.estimated_price == Decimal("450")
    assert result.description_generated is False
    assert "16" not in result.suggested_description
    assert "RAM" not in result.suggested_description
    assert "Dell XPS" in result.suggested_description


def test_input_validation_and_search_failure() -> None:
    with pytest.raises(SellerAssistantValidationError):
        SellerProductInput.from_mapping({"title": "Laptop", "sellerPrice": 0, "currency": "MAD"})
    with pytest.raises(SellerAssistantValidationError):
        SellerProductInput.from_mapping({"title": "Laptop", "sellerPrice": 10, "currency": "dirham"})
    search = Mock()
    search.search.side_effect = RuntimeError("down")
    with pytest.raises(SellerAssistantUnavailableError):
        SellerAssistantService(search).assist(_seller())


def test_generic_lexical_overlap_is_not_a_comparable(product: Product) -> None:
    search = Mock()
    search.search.return_value = SearchResult(
        "unknown scientific instrument",
        (
            SearchResultItem(
                replace(product, title="produit test"),
                0.82,
                relevance_reason="focal_lexical_evidence",
                lexical_terms=("produit",),
                match_type="relevant",
            ),
        ),
    )
    result = SellerAssistantService(search).assist(
        _seller(title="unknown scientific instrument", description="special product")
    )
    assert result.comparables_count == 0
    assert result.confidence == "none"


def test_simple_estimate_exposes_mean_range_and_robust_suggestion(product: Product) -> None:
    search = Mock()
    prices = [350, 400, 420, 450, 500, 5000]
    search.search.return_value = _result(
        replace(product, title="Business laptop"), prices,
        titles=["Laptop notebook"] * len(prices),
    )

    result = SellerAssistantService(search).estimate_price("Laptop Dell XPS")

    assert result.suggested_price == Decimal("420")
    assert result.mean_price == sum(map(Decimal, prices)) / Decimal(6)
    assert result.minimum == Decimal("350")
    assert result.maximum == Decimal("5000")
    assert len(result.comparables) == 6
    search.search.assert_called_once_with(
        "Laptop Dell XPS", None, source=SearchSource.TEXT, include_all=True
    )


def test_estimate_and_check_share_full_catalogue_comparable_selection(
    product: Product,
) -> None:
    search = Mock()
    exact = replace(
        product, id="exact", title="Dell laptop XPS", brand="Dell",
        category="Électronique", price=Decimal("400"), status="ACCEPTED",
    )
    alternative = replace(
        product, id="alternative", title="HP laptop", brand="HP",
        category="Électronique", price=Decimal("600"), status="PENDING",
    )
    accessory = replace(
        product, id="accessory", title="Laptop messenger bag", brand="Adidas",
        category="Femmes", price=Decimal("80"), status="SOLD", is_sold=True,
    )
    unrelated = replace(
        product, id="unrelated", title="Luxury perfume", brand="Dior",
        category="Femmes", price=Decimal("900"),
    )
    search.search.return_value = SearchResult(
        "Dell laptop",
        (
            SearchResultItem(exact, .90, "focal_lexical_evidence", ("laptop",), "exact"),
            SearchResultItem(alternative, .84, "product_family:computers", (), "similar"),
            SearchResultItem(accessory, .79, "product_domain:computers", (), "broad_similar"),
            SearchResultItem(unrelated, .78, "full_catalog", (), "unrelated"),
        ),
        structured_query=StructuredSearchQuery(
            original_query="Dell laptop", semantic_query="Dell laptop",
            language=SupportedLanguage.ENGLISH, intent=QueryIntent.PRODUCT_SEARCH,
            brand="Dell", product_type_hint="laptop",
        ),
        faiss_candidates_count=123,
        candidate_products_count=123,
        total_catalog_products=123,
    )
    service = SellerAssistantService(search)

    estimate = service.estimate_price("Dell laptop")
    check = service.check_price("Dell laptop", 450)

    assert [item.product.id for item in estimate.comparables] == [
        "exact", "alternative", "accessory"
    ]
    assert [item.product.id for item in check.comparables] == [
        "exact", "alternative", "accessory"
    ]
    assert estimate.candidate_products_count == check.candidate_products_count == 123
    assert estimate.total_catalog_products == check.total_catalog_products == 123
    assert search.search.call_args_list == [
        call("Dell laptop", None, source=SearchSource.TEXT, include_all=True),
        call("Dell laptop", None, source=SearchSource.TEXT, include_all=True),
    ]


@pytest.mark.parametrize(
    ("price", "alert"),
    [(100, "too_low"), (300, "low"), (400, "fair"), (500, "high"), (700, "too_high")],
)
def test_simple_price_check_uses_quartiles(
    product: Product, price: int, alert: str
) -> None:
    search = Mock()
    prices = [300, 350, 400, 450, 500]
    search.search.return_value = _result(
        replace(product, title="Laptop notebook"), prices,
        titles=["Laptop notebook"] * len(prices),
    )

    result = SellerAssistantService(search).check_price("Laptop", price)

    assert result.alert == alert
    assert result.stats is not None
    assert result.stats.median == Decimal("400")
    assert result.stats.p25 == Decimal("350")
    assert result.stats.p75 == Decimal("450")
    assert result.comparables_count == 5


def test_simple_description_without_image_has_grounded_fallback() -> None:
    provider = Mock()
    provider.suggest_seller_description.side_effect = RuntimeError("down")

    description, generated = SellerAssistantService(
        Mock(), provider
    ).suggest_description("Dell XPS", "Électronique", "portable")

    assert generated is False
    assert "Dell XPS" in description
    assert "portable" in description
    assert "RAM" not in description
