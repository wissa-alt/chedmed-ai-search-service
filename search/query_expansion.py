"""Deterministic query expansion backed by the ChedMed synonym resource."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from search.product_type_resolver import ProductTypeResolver
from search.synonym_resources import (
    SynonymResourceError,
    load_synonym_groups,
    normalize_synonym_text,
)

LOGGER = logging.getLogger(__name__)


class QueryExpansionError(ValueError):
    """Raised when a query cannot be expanded safely."""


class QueryExpansionService:
    """Expand natural-language queries with explicit synonym aliases.

    The original query is always preserved.

    Category aliases are used only when a category is explicitly detected.
    Official ChedMed category labels are never injected into the semantic
    query.

    Product type is kept separate from category. If the synonym resource
    contains product-type information, it can be handled independently by
    the caller/resource design.
    """

    def __init__(
        self,
        synonyms_file: str | Path | None = None,
        max_aliases_per_category: int = 3,
        max_aliases_per_product_type: int = 3,
    ) -> None:
        """Load validated synonym groups from the repository resource.

        Args:
            synonyms_file:
                Explicit synonym resource path, or the default resource.

            max_aliases_per_category:
                Maximum number of aliases added for one category.

        Raises:
            QueryExpansionError:
                If the configuration or synonym resource is invalid.
        """

        if (
            not isinstance(max_aliases_per_category, int)
            or isinstance(max_aliases_per_category, bool)
            or max_aliases_per_category <= 0
        ):
            raise QueryExpansionError(
                "max_aliases_per_category doit être un entier positif."
            )
        if (
            not isinstance(max_aliases_per_product_type, int)
            or isinstance(max_aliases_per_product_type, bool)
            or max_aliases_per_product_type <= 0
        ):
            raise QueryExpansionError(
                "max_aliases_per_product_type doit être un entier positif."
            )

        try:
            self._groups = load_synonym_groups(synonyms_file)
        except SynonymResourceError as exc:
            raise QueryExpansionError(
                "Impossible de charger les synonymes de requête."
            ) from exc

        self._max_aliases_per_category = max_aliases_per_category
        self._max_aliases_per_product_type = max_aliases_per_product_type
        self._product_types = ProductTypeResolver()

        self._synonyms_path = (
            Path(synonyms_file).expanduser()
            if synonyms_file is not None
            else None
        )

        self._alias_count = sum(
            len(aliases)
            for aliases in self._groups.values()
        )

        LOGGER.info(
            "Dictionnaire de synonymes chargé : %d alias, %d catégories.",
            self._alias_count,
            len(self._groups),
        )

    def expand(
        self,
        query: str,
        product_type: str | None = None,
        category: str | None = None,
        include_category_aliases: bool = True,
    ) -> str:
        """Expand a natural-language query using relevant synonym aliases.

        Args:
            query:
                Original user query.

            product_type:
                Semantic product type extracted by the query-understanding
                layer, for example ``"sberdila"``.

            category:
                Semantic category extracted by the query-understanding layer,
                for example ``"rjal"``.

        Returns:
            The original query followed by relevant synonym aliases.

        Important:
            The official ChedMed category is never added to the embedding
            query.

        Example:

            ``bghit sberdila dial rjal``

        remains the beginning of the query. If ``rjal`` is an alias of the
        official ``Hommes`` category, aliases associated with that category
        may be added, but ``Hommes`` itself is never injected.
        """

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        if not isinstance(query, str):
            raise QueryExpansionError(
                "query doit être une chaîne de caractères."
            )

        if product_type is not None and not isinstance(
            product_type,
            str,
        ):
            raise QueryExpansionError(
                "product_type doit être une chaîne de caractères."
            )

        if category is not None and not isinstance(
            category,
            str,
        ):
            raise QueryExpansionError(
                "category doit être une chaîne de caractères."
            )
        if not isinstance(include_category_aliases, bool):
            raise QueryExpansionError(
                "include_category_aliases doit être un booléen."
            )

        # ---------------------------------------------------------
        # Normalize input values
        # ---------------------------------------------------------

        original_query = " ".join(query.split())

        if not original_query:
            raise QueryExpansionError(
                "query ne peut pas être vide."
            )

        normalized_query = normalize_synonym_text(
            original_query
        )

        normalized_category = (
            normalize_synonym_text(category)
            if category is not None
            else None
        )

        expansions: list[str] = []

        # ---------------------------------------------------------
        # CATEGORY EXPANSION
        # ---------------------------------------------------------
        #
        # Example:
        #
        # category = "rjal"
        #
        # resource:
        #
        # "Hommes": [
        #     "homme",
        #     "hommes",
        #     "rjal",
        #     ...
        # ]
        #
        # We must identify "Hommes" through its aliases.
        #
        # We MUST NOT do:
        #
        #     official_category == normalized_category
        #
        # because:
        #
        #     "hommes" != "rjal"
        #
        # Instead, we check whether the requested category is
        # one of the aliases of the official category.
        # ---------------------------------------------------------

        # Explicit hints are useful, but expansion also remains usable as a
        # standalone semantic service by discovering complete aliases in text.
        if include_category_aliases and not normalized_category:
            for official_category, aliases in self._groups.items():
                if any(_contains_complete_phrase(normalized_query, alias) for alias in aliases):
                    normalized_category = normalize_synonym_text(official_category)
                    break

        if include_category_aliases and normalized_category:

            for official_category, aliases in self._groups.items():

                normalized_official_category = (
                    normalize_synonym_text(
                        official_category
                    )
                )

                normalized_aliases = {
                    normalize_synonym_text(alias)
                    for alias in aliases
                }

                category_matches = (
                    normalized_category
                    == normalized_official_category
                    or normalized_category
                    in normalized_aliases
                )

                if not category_matches:
                    continue

                LOGGER.debug(
                    "Catégorie sémantique %r associée au groupe %r.",
                    category,
                    official_category,
                )

                matched_aliases = [
                    alias
                    for alias in aliases
                    if _contains_complete_phrase(
                        normalized_query,
                        alias,
                    )
                ]

                if matched_aliases:
                    expansions.extend(
                        self._select_expansions(
                            official_category=official_category,
                            aliases=aliases,
                            normalized_query=normalized_query,
                            matched_aliases=matched_aliases,
                        )
                    )

                # One category group is enough.
                break

        # ---------------------------------------------------------
        # PRODUCT TYPE
        # ---------------------------------------------------------
        #
        # product_type is intentionally NOT mixed with category
        # aliases.
        #
        # Example:
        #
        # product_type = "sberdila"
        #
        # We do not want:
        #
        #     sberdila
        #     casquette
        #     hoodie
        #     montre
        #
        # simply because they belong to "Hommes".
        #
        # Product-type synonym expansion requires dedicated
        # product-type groups in the synonym resource.
        # ---------------------------------------------------------

        resolved_product_type = self._product_types.resolve(product_type)
        if resolved_product_type is None:
            for candidate in (original_query.split() + [original_query]):
                resolved_product_type = self._product_types.resolve(candidate)
                if resolved_product_type:
                    break
        if resolved_product_type:
            added = 0
            for alias in self._product_types.aliases_for(resolved_product_type):
                if not _contains_complete_phrase(normalized_query, alias):
                    expansions.append(alias)
                    added += 1
                    if added >= self._max_aliases_per_product_type:
                        break

        # ---------------------------------------------------------
        # Build final query
        # ---------------------------------------------------------

        result = _join_unique_phrases(
            [
                original_query,
                *expansions,
            ]
        )

        LOGGER.debug(
            "Query expansion applied: %r -> %r",
            original_query,
            result,
        )

        return result

    def _select_expansions(
        self,
        official_category: str,
        aliases: tuple[str, ...],
        normalized_query: str,
        matched_aliases: list[str],
    ) -> list[str]:
        """Select safe aliases without injecting the official category.

        Already-present aliases are excluded to avoid unnecessary
        duplication.

        The official category itself is explicitly excluded.
        """

        selected: list[str] = []

        normalized_official_category = normalize_synonym_text(
            official_category
        )

        for alias in (
            *matched_aliases,
            *aliases,
        ):

            normalized_alias = normalize_synonym_text(
                alias
            )

            if not normalized_alias:
                continue

            # Never inject the official ChedMed category.
            if normalized_alias == normalized_official_category:
                continue

            # Do not repeat an alias already present in the query.
            if _contains_complete_phrase(
                normalized_query,
                alias,
            ):
                continue

            # Avoid duplicate expansions.
            if any(
                normalize_synonym_text(existing)
                == normalized_alias
                for existing in selected
            ):
                continue

            selected.append(alias)

            if (
                len(selected)
                >= self._max_aliases_per_category
            ):
                break

        return selected


def _contains_complete_phrase(
    normalized_query: str,
    alias: str,
) -> bool:
    """Return whether an alias occurs as a complete word or phrase."""

    normalized_alias = normalize_synonym_text(alias)

    if not normalized_alias:
        return False

    pattern = (
        rf"(?<!\w)"
        rf"{re.escape(normalized_alias)}"
        rf"(?!\w)"
    )

    return re.search(
        pattern,
        normalized_query,
    ) is not None


def _join_unique_phrases(
    phrases: list[str],
) -> str:
    """Join phrases while removing normalized duplicates."""

    unique: list[str] = []
    seen: set[str] = set()

    for phrase in phrases:

        normalized = normalize_synonym_text(
            phrase
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        unique.append(phrase)
        seen.add(normalized)

    return " ".join(unique)
