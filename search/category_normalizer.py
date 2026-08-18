"""Deterministic normalisation of semantic categories to ChedMed categories."""

from __future__ import annotations

import logging
from pathlib import Path

from search.synonym_resources import (
    SynonymResourceError,
    load_synonym_groups,
    normalize_synonym_text,
)

LOGGER = logging.getLogger(__name__)


class CategoryNormalizationError(RuntimeError):
    """Raised when a category cannot be safely normalised."""


CategoryNormalizerError = CategoryNormalizationError


class CategoryNormalizer:
    """Map only explicit synonym-resource aliases to official categories.

    Matching is deliberately exact after text normalisation.  A conservative
    terminal-plural fallback is attempted only when that singular form is also
    an explicit alias in the resource; arbitrary substring matching is never
    used.
    """

    def __init__(self, synonyms_path: str | Path | None = None) -> None:
        """Load one immutable synonym view from the configured JSON resource."""
        try:
            self._groups = load_synonym_groups(synonyms_path)
        except SynonymResourceError as exc:
            raise CategoryNormalizationError(
                "Impossible de charger les synonymes de catégories ChedMed."
            ) from exc

        self._official_by_normalized = {
            normalize_synonym_text(category): category
            for category in self._groups
        }
        self._aliases = {
            normalize_synonym_text(alias): category
            for category, aliases in self._groups.items()
            for alias in aliases
        }

    def normalize(self, category: str | None) -> str | None:
        """Return an official category for an explicit semantic category.

        Args:
            category: Raw category extracted by the LLM, or ``None``.

        Returns:
            The official ChedMed category, or ``None`` when no reliable
            mapping exists.

        Raises:
            CategoryNormalizationError: If ``category`` has an invalid type.
        """
        if category is None:
            return None
        if not isinstance(category, str):
            raise CategoryNormalizationError(
                "La catégorie doit être une chaîne de caractères ou None."
            )

        normalized = normalize_synonym_text(category)
        if not normalized:
            return None

        official = self._official_by_normalized.get(normalized)
        if official is not None:
            LOGGER.debug("Catégorie ChedMed déjà officielle : %s", official)
            return official

        mapped = self._aliases.get(normalized)
        if mapped is None:
            for candidate in _singular_candidates(normalized):
                mapped = self._aliases.get(candidate)
                if mapped is not None:
                    break

        if mapped is None:
            LOGGER.warning("Unknown category: %s", category)
            return None

        LOGGER.debug("Catégorie normalisée : %r -> %r", category, mapped)
        return mapped

    def matches(
        self,
        requested_category: str | None,
        product_category: str | None,
    ) -> bool:
        """Return whether both values resolve to the same official category."""
        if not isinstance(requested_category, str) or not isinstance(product_category, str):
            return False
        requested = self.normalize(requested_category)
        product = self.normalize(product_category)
        return requested is not None and requested == product

    def categories(self) -> tuple[str, ...]:
        """Return official ChedMed categories from the external resource."""
        return tuple(self._groups)

    def get_supported_categories(self) -> tuple[str, ...]:
        """Return the backward-compatible alias of :meth:`categories`."""
        return self.categories()

    def is_valid_chedmed_category(self, category: str | None) -> bool:
        """Return whether a value is an exact official ChedMed category."""
        return isinstance(category, str) and (
            normalize_synonym_text(category) in self._official_by_normalized
        )


def _singular_candidates(normalized_category: str) -> tuple[str, ...]:
    """Return only conservative singular candidates for an exact lookup."""
    if len(normalized_category) > 3 and normalized_category.endswith("s"):
        return (normalized_category[:-1],)
    return ()
