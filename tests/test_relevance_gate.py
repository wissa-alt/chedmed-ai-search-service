"""Tests for relative post-ranking relevance gating."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from models.product import Product
from models.query_analysis import QueryIntent, SupportedLanguage
from models.search_query import StructuredSearchQuery
from search.ranking_engine import RankingCandidate, RankingEngine
from search.relevance_gate import RelevanceGate


def _product(identifier: str, title: str, category: str = "Hommes") -> Product:
    return Product(
        id=identifier, title=title, description=title, category=category,
        brand=None, color=None, condition=None, price=Decimal("100"),
        currency="MAD", city=None, image_urls=(), status="ACTIVE", is_sold=False,
        updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def _query(text: str, **values: object) -> StructuredSearchQuery:
    return StructuredSearchQuery(
        original_query=text, semantic_query=text, language=SupportedLanguage.FRENCH,
        intent=QueryIntent.PRODUCT_SEARCH, **values,
    )


def _gate() -> RelevanceGate:
    return RelevanceGate(RankingEngine())


def test_strong_candidate_survives_while_noise_does_not_fill_top_k() -> None:
    candidates = [
        RankingCandidate(_product("120", "Perfumes"), 0.849),
        RankingCandidate(_product("noise-1", "Test Product"), 0.806),
        RankingCandidate(_product("noise-2", "Generic Item"), 0.801),
    ]
    relevant, _ = _gate().apply(candidates, _query("parfum", product_type_hint="parfum"))
    assert [item.product.id for item in relevant[:10]] == ["120"]
    assert len(relevant[:10]) == 1


def test_no_relevant_product_after_category_filter_returns_empty() -> None:
    candidates = [
        RankingCandidate(_product("86", "Women bag", "Femmes"), 0.820),
        RankingCandidate(_product("139", "Abc product", "Femmes"), 0.818),
        RankingCandidate(_product("98", "Polarized sunglasses", "Femmes"), 0.782),
    ]
    query = _query(
        "bghit parfum pour femme", category="Femmes", category_hint="femmes",
        product_type_hint="parfum",
    )
    relevant, decisions = _gate().apply(candidates, query)
    assert relevant == []
    assert all(not decision.accepted for decision in decisions)


def test_isolated_semantic_leader_passes_without_lexical_alias() -> None:
    candidates = [
        RankingCandidate(_product("1", "Black Peshawari Chappal"), 0.828),
        RankingCandidate(_product("51", "Hoodie Maroc"), 0.807),
    ]
    query = _query(
        "bghit sberdila dial rjal", category="Hommes", category_hint="rjal",
        product_type="chaussures", product_type_hint="sberdila",
    )
    relevant, decisions = _gate().apply(candidates, query)
    assert [item.product.id for item in relevant] == ["1"]
    assert decisions[0].lexical_terms == ()
    assert decisions[0].reason == "isolated_semantic_leader"


def test_leader_margin_is_configurable() -> None:
    ranking = RankingEngine()
    candidates = [
        RankingCandidate(_product("1", "Unknown semantic match"), 0.820),
        RankingCandidate(_product("2", "Other item"), 0.810),
    ]
    query = _query("untranslated concept")
    assert RelevanceGate(ranking).apply(candidates, query)[0] == []
    relevant, _ = RelevanceGate(ranking, leader_margin=0.005).apply(candidates, query)
    assert [item.product.id for item in relevant] == ["1"]

