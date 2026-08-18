"""Tests for non-destructive ranking on top of FAISS similarity."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from models.product import Product
from models.query_analysis import QueryIntent, SupportedLanguage
from models.search_query import StructuredSearchQuery
from search.ranking_engine import RankingCandidate, RankingEngine


def _product(identifier: str, title: str) -> Product:
    return Product(
        id=identifier, title=title, description="", category="Hommes",
        brand=None, color=None, condition=None, price=Decimal("100"),
        currency="MAD", city=None, image_urls=(), status="ACTIVE", is_sold=False,
        updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def test_missing_product_type_alias_has_no_penalty_or_exclusion() -> None:
    engine = RankingEngine()
    query = StructuredSearchQuery(
        original_query="sberdila", semantic_query="sberdila", language=SupportedLanguage.DARIJA,
        intent=QueryIntent.PRODUCT_SEARCH, category="Hommes", product_type="chaussures",
    )
    chappal = RankingCandidate(_product("1", "Black Peshawari Chappal"), 0.90)
    shoes = RankingCandidate(_product("2", "Shoes"), 0.81)

    ranked = engine.rank([chappal, shoes], query)

    assert {item.product.id for item in ranked} == {"1", "2"}
    assert chappal.semantic_score == 0.90
    assert engine.score(chappal, query).product_type_bonus == 0.0
    assert engine.score(shoes, query).product_type_bonus > 0.0
