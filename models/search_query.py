"""Canonical query passed through the ChedMed search pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from models.query_analysis import QueryIntent, SupportedLanguage


class SearchSource(str, Enum):
    """Origin of the natural-language text entering the shared pipeline."""

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class StructuredSearchQuery:
    """Separate semantic retrieval text from resolved business filters."""

    original_query: str
    semantic_query: str
    language: SupportedLanguage
    intent: QueryIntent
    source: SearchSource = SearchSource.TEXT
    category: str | None = None
    product_type: str | None = None
    brand: str | None = None
    color: str | None = None
    city: str | None = None
    condition: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    currency: str | None = None
    is_new: bool | None = None
    is_used: bool | None = None
    is_sold: bool | None = None
    confidence: float = 0.0
    category_hint: str | None = None
    product_type_hint: str | None = None

    @property
    def has_filters(self) -> bool:
        """Return whether at least one applicable resolved filter exists."""
        return any(
            (
                self.category,
                self.brand,
                self.color,
                self.city,
                self.condition,
                self.min_price is not None,
                self.max_price is not None,
                self.is_new is not None,
                self.is_used is not None,
                self.is_sold is not None,
            )
        )
