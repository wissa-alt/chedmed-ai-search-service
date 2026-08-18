"""Explicit final ranking for already retrieved and filtered products."""

from __future__ import annotations

from dataclasses import dataclass

from models.product import Product
from models.search_query import StructuredSearchQuery
from search.product_filter import _normalise_text
from search.product_type_resolver import ProductTypeResolver


@dataclass(frozen=True, slots=True)
class RankingCandidate:
    product: Product
    semantic_score: float


@dataclass(frozen=True, slots=True)
class RankingScore:
    """Explainable ranking score without changing acceptance semantics."""

    total: float
    semantic: float
    product_type_bonus: float
    category_bonus: float


class RankingEngine:
    """Combine semantic order with small, explainable lexical bonuses."""

    def __init__(self, product_types: ProductTypeResolver | None = None) -> None:
        self._product_types = product_types or ProductTypeResolver()

    def rank(
        self,
        candidates: list[RankingCandidate],
        query: StructuredSearchQuery,
    ) -> list[RankingCandidate]:
        """Sort candidates without replacing their original FAISS score."""
        return sorted(
            candidates,
            key=lambda item: self.score(item, query).total,
            reverse=True,
        )

    def score(self, item: RankingCandidate, query: StructuredSearchQuery) -> RankingScore:
        """Return the semantic score and bounded positive bonuses."""
        text = _normalise_text(f"{item.product.title} {item.product.description}")
        product_type_bonus = 0.02 if query.product_type and any(
            _normalise_text(alias) in text
            for alias in self._product_types.aliases_for(query.product_type)
        ) else 0.0
        category_bonus = 0.01 if (
            query.category
            and _normalise_text(item.product.category) == _normalise_text(query.category)
        ) else 0.0
        return RankingScore(
            total=item.semantic_score + product_type_bonus + category_bonus,
            semantic=item.semantic_score,
            product_type_bonus=product_type_bonus,
            category_bonus=category_bonus,
        )
