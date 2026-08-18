"""Domain models describing the semantic understanding of a user query."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(str, Enum):
    """High-level user intent inferred from the query."""

    PRODUCT_SEARCH = "product_search"
    PRODUCT_COMPARISON = "product_comparison"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    UNKNOWN = "unknown"


class SupportedLanguage(str, Enum):
    """Languages explicitly supported by the query understanding layer."""

    FRENCH = "fr"
    ENGLISH = "en"
    ARABIC = "ar"
    DARIJA = "darija"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class SearchFilters:
    """Structured filters extracted from a natural-language query.

    Category values live here because they are filtering inputs:

    - ``category_raw`` is the semantic value returned by the LLM.
    - ``category_normalized`` is the official ChedMed category calculated by
      :class:`search.category_normalizer.CategoryNormalizer`.

    The two values must never be substituted for one another.
    """

    category_raw: str | None = None
    category_normalized: str | None = None

    product_type_raw: str | None = None
    product_type_normalized: str | None = None


    brand: str | None = None

    city: str | None = None
    color: str | None = None
    condition: str | None = None

    min_price: float | None = None
    max_price: float | None = None

    currency: str | None = None

    is_new: bool | None = None
    is_used: bool | None = None
    is_sold: bool | None = None


@dataclass(slots=True, init=False)
class QueryAnalysis:
    """Complete semantic representation of a search query.

    ``SearchFilters`` is the canonical owner of category data.  The
    ``category_raw`` and ``category_normalized`` properties below remain as
    compatibility views for callers that previously read them on this model.
    """

    original_query: str

    language: SupportedLanguage

    intent: QueryIntent

    expanded_query: str

    keywords: list[str]
    filters: SearchFilters
    confidence: float
    reasoning: str | None

    def __init__(
        self,
        original_query: str,
        language: SupportedLanguage,
        intent: QueryIntent,
        expanded_query: str,
        category_raw: str | None = None,
        category_normalized: str | None = None,
        keywords: list[str] | None = None,
        filters: SearchFilters | None = None,
        confidence: float = 0.0,
        reasoning: str | None = None,
    ) -> None:
        """Create an analysis while preserving the former category arguments.

        Category data is stored only in ``filters``.  Legacy category keyword
        arguments are accepted and forwarded to that canonical location.
        Conflicting values are rejected rather than silently choosing one.
        """
        self.original_query = original_query
        self.language = language
        self.intent = intent
        self.expanded_query = expanded_query
        self.keywords = list(keywords or [])
        self.filters = filters if filters is not None else SearchFilters()
        self.confidence = confidence
        self.reasoning = reasoning
        self._merge_legacy_categories(category_raw, category_normalized)

    @property
    def category_raw(self) -> str | None:
        """Return the LLM category retained in the canonical filter object."""
        return self.filters.category_raw

    @category_raw.setter
    def category_raw(self, value: str | None) -> None:
        """Set the canonical raw category through the compatibility view."""
        self.filters.category_raw = value

    @property
    def category_normalized(self) -> str | None:
        """Return the normalized category retained in the filter object."""
        return self.filters.category_normalized

    @category_normalized.setter
    def category_normalized(self, value: str | None) -> None:
        """Set the canonical normalized category through the compatibility view."""
        self.filters.category_normalized = value

    def _merge_legacy_categories(
        self,
        category_raw: str | None,
        category_normalized: str | None,
    ) -> None:
        """Merge accepted legacy constructor values into ``SearchFilters``."""
        if (
            category_raw is not None
            and self.filters.category_raw is not None
            and category_raw != self.filters.category_raw
        ):
            raise ValueError("category_raw est incohérente avec les filtres.")
        if (
            category_normalized is not None
            and self.filters.category_normalized is not None
            and category_normalized != self.filters.category_normalized
        ):
            raise ValueError("category_normalized est incohérente avec les filtres.")
        if category_raw is not None:
            self.filters.category_raw = category_raw
        if category_normalized is not None:
            self.filters.category_normalized = category_normalized

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the analysis.

        Both category values are exposed separately so that logs,
        debugging and monitoring can distinguish the LLM output from
        the normalized ChedMed category.
        """

        return {
            "original_query": self.original_query,

            "language": self.language.value,

            "intent": self.intent.value,

            "expanded_query": self.expanded_query,

            "category_raw": self.category_raw,

            "category_normalized": self.category_normalized,

            "keywords": self.keywords,

            "filters": {
                "category_raw": self.filters.category_raw,
                "category_normalized": self.filters.category_normalized,
                "product_type_raw": self.filters.product_type_raw,
                "product_type_normalized": self.filters.product_type_normalized,
                "brand": self.filters.brand,
                "city": self.filters.city,
                "color": self.filters.color,
                "condition": self.filters.condition,
                "min_price": self.filters.min_price,
                "max_price": self.filters.max_price,
                "currency": self.filters.currency,
                "is_new": self.filters.is_new,
                "is_used": self.filters.is_used,
                "is_sold": self.filters.is_sold,
            },

            "confidence": self.confidence,

            "reasoning": self.reasoning,
        }

    @classmethod
    def empty(cls, query: str) -> "QueryAnalysis":
        """
        Return a fallback analysis when semantic understanding fails.

        Category values remain None because no reliable category
        information was produced.
        """

        return cls(
            original_query=query,
            language=SupportedLanguage.UNKNOWN,
            intent=QueryIntent.UNKNOWN,
            expanded_query=query,
        )
