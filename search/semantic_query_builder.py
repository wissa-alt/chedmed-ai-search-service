"""Build focused retrieval text from product concepts and resolved constraints."""

from __future__ import annotations

import re
from typing import Protocol
from dataclasses import dataclass

from models.search_query import SearchSource, StructuredSearchQuery
from search.product_type_resolver import ProductTypeResolver
from search.synonym_resources import load_synonym_groups, normalize_synonym_text


@dataclass(frozen=True, slots=True)
class FocusedSemanticQuery:
    """Focused text plus an audit trail of removed structured expressions."""

    text: str
    removed_constraints: tuple[str, ...] = ()


class ExpansionPort(Protocol):
    def expand(
        self,
        query: str,
        product_type: str | None = None,
        category: str | None = None,
        include_category_aliases: bool = True,
    ) -> str: ...


class SemanticQueryBuilder:
    """Separate semantic product meaning from reliable structured filters."""

    def __init__(self, expansion: ExpansionPort) -> None:
        self._expansion = expansion
        self._product_types = ProductTypeResolver()
        self._category_groups = load_synonym_groups()

    def build(self, query: StructuredSearchQuery) -> FocusedSemanticQuery:
        """Return product-focused text without trusting an LLM rewrite."""
        if query.source is SearchSource.IMAGE:
            expanded = self._expansion.expand(
                query.original_query,
                product_type=query.product_type,
                category=None,
                include_category_aliases=False,
            )
            return FocusedSemanticQuery(expanded)

        product_anchor = self._product_anchor(query)
        if product_anchor:
            expanded = self._expansion.expand(
                product_anchor,
                product_type=query.product_type,
                category=None,
                include_category_aliases=False,
            )
            removed = self._structured_constraints_present(query)
            return FocusedSemanticQuery(expanded, removed)

        focused = query.original_query
        removed: list[str] = []
        # Do not strip category aliases blindly here. The current category
        # resource also contains product-bearing terms (for example devices
        # under Electronique). Without a validated product anchor, retaining
        # an ambiguous term is safer than deleting the user's core concept.

        # These values are retained as strict filters only when the exact
        # provider-extracted value occurs in the original query. Brand is
        # intentionally preserved because there is no deterministic brand
        # resolver and unknown/new brands remain useful semantic concepts.
        for value in (query.city, query.color, query.condition):
            if value:
                focused, value_removed = _remove_phrases(focused, (value,))
                removed.extend(value_removed)

        focused = _clean(focused) or query.original_query
        expanded = self._expansion.expand(
            focused,
            product_type=None,
            category=None,
            include_category_aliases=False,
        )
        return FocusedSemanticQuery(expanded, tuple(dict.fromkeys(removed)))

    def _structured_constraints_present(
        self, query: StructuredSearchQuery
    ) -> tuple[str, ...]:
        """Audit only reliable structured expressions present in source text."""
        candidates: list[str] = []
        if query.category:
            candidates.extend(self._category_groups.get(query.category, ()))
        candidates.extend(
            value
            for value in (query.city, query.color, query.condition)
            if value
        )
        matches = [
            value for value in candidates
            if _contains_phrase(query.original_query, value)
        ]
        return tuple(dict.fromkeys(matches))

    def _product_anchor(self, query: StructuredSearchQuery) -> str | None:
        """Return only a product alias validated against the original text."""
        if not query.product_type:
            return None
        candidates: list[str] = []
        if (
            query.product_type_hint
            and self._product_types.resolve(query.product_type_hint) == query.product_type
        ):
            candidates.append(query.product_type_hint)
        candidates.extend(self._product_types.aliases_for(query.product_type))
        matches = [
            candidate
            for candidate in candidates
            if _contains_phrase(query.original_query, candidate)
        ]
        if not matches:
            return None
        return max(matches, key=lambda value: len(normalize_synonym_text(value)))


def _remove_phrases(text: str, phrases: tuple[str, ...]) -> tuple[str, list[str]]:
    result = text
    removed: list[str] = []
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = _phrase_pattern(phrase)
        if re.search(pattern, result, flags=re.IGNORECASE):
            result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
            removed.append(phrase)
    return _clean(result), removed


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(_phrase_pattern(phrase), text, flags=re.IGNORECASE) is not None


def _phrase_pattern(phrase: str) -> str:
    words = phrase.strip().split()
    body = r"\s+".join(re.escape(word) for word in words)
    return rf"(?<!\w){body}(?!\w)"


def _clean(value: str) -> str:
    return " ".join(value.strip(" ,;:-").split())
