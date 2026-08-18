"""Apply only resolved structured filters to catalogue product snapshots."""

from __future__ import annotations

from models.product import Product
from models.query_analysis import SearchFilters
from models.search_query import SearchSource, StructuredSearchQuery
from search.product_filter import ProductFilterService


class FilterEngine:
    """Deterministic post-retrieval business filtering."""

    def __init__(self, product_filter: ProductFilterService | None = None) -> None:
        self._base = product_filter or ProductFilterService()

    def matches(self, product: Product, query: StructuredSearchQuery) -> bool:
        """Return True when all and only resolved filters match."""
        matches, _ = self.evaluate(product, query)
        return matches

    def evaluate(
        self,
        product: Product,
        query: StructuredSearchQuery,
    ) -> tuple[bool, str | None]:
        """Return a decision based only on fields represented by Product."""
        image_observation = query.source is SearchSource.IMAGE
        filters = SearchFilters(
            category_normalized=query.category,
            brand=query.brand,
            city=None if image_observation else query.city,
            color=None if image_observation else query.color,
            condition=None if image_observation else query.condition,
            min_price=None if image_observation else query.min_price,
            max_price=None if image_observation else query.max_price,
            currency=None if image_observation else query.currency,
            is_new=None if image_observation else query.is_new,
            is_used=None if image_observation else query.is_used,
            is_sold=query.is_sold,
        )
        return self._base.match_with_reason(product, filters)
