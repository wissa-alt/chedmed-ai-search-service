
"""
Deterministic product filtering for semantic search results.

This module applies the structured filters extracted from the user's query
to current ChedMed Product snapshots.

Responsibilities:
    - Filter products by brand.
    - Filter products by category.
    - Filter products by city.
    - Filter products by color.
    - Filter products by condition.
    - Filter products by price range.
    - Filter products by currency.
    - Filter products by new/used status.
    - Filter products by sold status.

This service does NOT:
    - call FAISS;
    - call Groq;
    - call the ChedMed API;
    - generate embeddings;
    - modify products.

It is intentionally deterministic.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Iterable
from decimal import Decimal

from models.product import Product
from models.query_analysis import SearchFilters

LOGGER = logging.getLogger(__name__)

# Catalogue sentinel values are metadata states, not reliable evidence of a
# commercial-brand conflict.
_UNRELIABLE_BRAND_VALUES = frozenset(
    {
        "sans marque",
        "no brand",
        "unbranded",
        "unknown",
        "inconnu",
        "generic",
        "generique",
        "n a",
        "na",
    }
)


class ProductFilterError(RuntimeError):
    """Raised when product filtering cannot be completed safely."""


class ProductFilterService:
    """
    Apply structured search filters to ChedMed products.

    The service never changes the input products. It only returns products
    that satisfy the requested filters.
    """

    def filter(
        self,
        products: Iterable[Product],
        filters: SearchFilters,
    ) -> list[Product]:
        """
        Filter products according to the supplied search filters.

        Args:
            products:
                Current product snapshots retrieved from ChedMed.

            filters:
                Structured filters produced by QueryUnderstandingService.

        Returns:
            A list containing only products matching all applicable filters.

        Raises:
            ProductFilterError:
                If the products collection or filters object is invalid.
        """
        if products is None:
            raise ProductFilterError(
                "La liste des produits ne peut pas être None."
            )

        if not isinstance(filters, SearchFilters):
            raise ProductFilterError(
                "Les filtres doivent être une instance de SearchFilters."
            )

        filtered_products: list[Product] = []

        total = 0

        for product in products:
            total += 1

            try:
                if self.matches(product, filters):
                    filtered_products.append(product)

            except Exception:
                LOGGER.exception(
                    "Impossible d'appliquer les filtres au produit %s.",
                    getattr(product, "id", "unknown"),
                )

        LOGGER.info(
            "Filtrage produits : %d/%d produit(s) conservé(s).",
            len(filtered_products),
            total,
        )

        return filtered_products

    def matches(
        self,
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """
        Return True if one product satisfies every active filter.

        All active filters are combined using logical AND.

        Args:
            product:
                Current ChedMed product.

            filters:
                Structured search filters.

        Returns:
            True when the product satisfies all active filters.
        """
        matches, _ = self.match_with_reason(product, filters)
        return matches

    def match_with_reason(
        self,
        product: Product,
        filters: SearchFilters,
    ) -> tuple[bool, str | None]:
        """Return the decision and the first reliable rejected field."""
        if product is None:
            return False, "missing_product"
        if not isinstance(filters, SearchFilters):
            raise ProductFilterError(
                "Les filtres doivent être une instance de SearchFilters."
            )
        checks = (
            ("brand", self._matches_brand),
            ("category", self._matches_category),
            ("city", self._matches_city),
            ("color", self._matches_color),
            ("condition", self._matches_condition),
            ("price", self._matches_price),
            ("currency", self._matches_currency),
            ("new_used", self._matches_new_used),
            ("is_sold", self._matches_sold),
        )
        for field_name, check in checks:
            if not check(product, filters):
                return False, field_name
        return True, None

    # ------------------------------------------------------------------
    # Text filters
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_brand(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """Reject only a verifiable conflict between explicit brands.

        Missing/generic catalogue metadata cannot disprove a brand visible in
        title or description. When metadata is inconclusive, semantic search
        keeps the candidate; an explicit different brand remains strict.
        """
        requested_brand = _normalise_text(filters.brand)

        if not requested_brand:
            return True

        product_text = _normalise_text(
            f"{getattr(product, 'title', '')} {getattr(product, 'description', '')}"
        )
        if _contains_phrase(product_text, requested_brand):
            return True

        product_brand = _normalise_text(
            getattr(product, "brand", None)
        )

        if not product_brand or product_brand in _UNRELIABLE_BRAND_VALUES:
            return True

        return (
            requested_brand == product_brand
            or requested_brand in product_brand
            or product_brand in requested_brand
        )


    @staticmethod
    def _matches_category(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """Match only the category normalized once after LLM processing.

        A raw but unresolved category deliberately skips strict category
        filtering.  This retains semantic-search recall for vocabulary absent
        from the current ChedMed synonym resource.
        """
        requested_category = _normalise_text(filters.category_normalized)
        if not requested_category:
            return True

        product_category = _normalise_text(getattr(product, "category", None))

        if not product_category:
            return False

        return requested_category == product_category

    @staticmethod
    def _matches_city(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """Check the requested city."""
        requested_city = _normalise_text(filters.city)

        if not requested_city:
            return True

        product_city = _normalise_text(
            getattr(product, "city", None)
        )

        if not product_city:
            return False

        return (
            requested_city == product_city
            or requested_city in product_city
            or product_city in requested_city
        )

    @staticmethod
    def _matches_color(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """
        Check the requested color.

        The comparison is intentionally textual. Translation between
        languages is handled upstream by query understanding and can be
        extended later with a dedicated color vocabulary.
        """
        requested_color = _normalise_text(filters.color)

        if not requested_color:
            return True

        product_color = _normalise_text(
            getattr(product, "color", None)
        )

        if not product_color:
            return False

        return (
            requested_color == product_color
            or requested_color in product_color
            or product_color in requested_color
        )

    @staticmethod
    def _matches_condition(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """Check the requested product condition."""
        requested_condition = _normalise_text(
            filters.condition
        )

        if not requested_condition:
            return True

        product_condition = _normalise_text(
            getattr(product, "condition", None)
        )

        if not product_condition:
            return False

        return (
            requested_condition == product_condition
            or requested_condition in product_condition
            or product_condition in requested_condition
        )

    # ------------------------------------------------------------------
    # Price and currency
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_price(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """
        Check minimum and maximum price constraints.

        If a price filter is active and the product does not have a valid
        numeric price, the product is rejected.

        Important:
            A semantic expression such as "pas cher" does not automatically
            create a price constraint. QueryUnderstandingService must provide
            min_price/max_price explicitly.
        """
        min_price = _parse_numeric(
            filters.min_price
        )

        max_price = _parse_numeric(
            filters.max_price
        )

        if min_price is None and max_price is None:
            return True

        product_price = _parse_numeric(
            getattr(product, "price", None)
        )

        if product_price is None:
            return False

        if min_price is not None and product_price < min_price:
            return False

        if max_price is not None and product_price > max_price:
            return False

        return True

    @staticmethod
    def _matches_currency(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """
        Check currency only when a price constraint is active.

        This avoids unnecessarily rejecting products when the query has no
        price constraint.
        """
        min_price = _parse_numeric(
            filters.min_price
        )

        max_price = _parse_numeric(
            filters.max_price
        )

        if min_price is None and max_price is None:
            return True

        requested_currency = _normalise_currency(
            filters.currency
        )

        if requested_currency is None:
            requested_currency = "MAD"

        product_currency = _normalise_currency(
            getattr(product, "currency", None)
        )

        if product_currency is None:
            return False

        return product_currency == requested_currency

    # ------------------------------------------------------------------
    # Product state
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_new_used(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """
        Check new/used constraints.

        Priority:
            1. Explicit is_new.
            2. Explicit is_used.
            3. Textual condition.
        """
        is_new = filters.is_new
        is_used = filters.is_used

        if is_new is True:
            if not _product_is_new(product):
                return False

        if is_used is True:
            if not _product_is_used(product):
                return False

        return True

    @staticmethod
    def _matches_sold(
        product: Product,
        filters: SearchFilters,
    ) -> bool:
        """
        Check sold status.

        The vector index should normally already exclude sold products.
        This method provides an additional safety layer.
        """
        requested_sold = filters.is_sold

        if requested_sold is None:
            return True

        product_sold = _parse_optional_bool(
            getattr(product, "is_sold", None)
        )

        if product_sold is None:
            return False

        return product_sold == requested_sold


# ======================================================================
# Helper functions
# ======================================================================


def _normalise_text(
    value: object,
) -> str:
    """
    Normalize text for deterministic comparisons.

    Operations:
        - None -> ""
        - convert to string
        - trim spaces
        - lowercase
        - remove accents
        - normalize internal whitespace
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    return " ".join(text.split())


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match a normalized brand as complete tokens, not a substring."""
    if not text or not phrase:
        return False
    words = phrase.split()
    body = r"\s+".join(re.escape(word) for word in words)
    return re.search(rf"(?<!\w){body}(?!\w)", text) is not None


def _normalise_currency(
    value: object,
) -> str | None:
    """Normalize common currency representations."""
    if value is None:
        return None

    normalized = _normalise_text(value)

    if not normalized:
        return None

    mapping = {
        "mad": "MAD",
        "dh": "MAD",
        "dhs": "MAD",
        "dirham": "MAD",
        "dirhams": "MAD",
        "mad marocain": "MAD",
        "eur": "EUR",
        "euro": "EUR",
        "euros": "EUR",
        "usd": "USD",
        "dollar": "USD",
        "dollars": "USD",
    }

    return mapping.get(
        normalized,
        normalized.upper(),
    )


def _parse_numeric(
    value: object,
) -> float | None:
    """
    Safely convert a value to a finite non-negative float.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float, Decimal)):
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            return None

        if numeric_value < 0:
            return None

        return numeric_value

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        try:
            numeric_value = float(
                text.replace(",", ".")
            )
        except ValueError:
            return None

        if not math.isfinite(numeric_value):
            return None

        if numeric_value < 0:
            return None

        return numeric_value

    return None


def _parse_optional_bool(
    value: object,
) -> bool | None:
    """Convert common boolean representations to bool or None."""
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = _normalise_text(value)

        if normalized in {
            "true",
            "1",
            "yes",
            "oui",
            "vrai",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "non",
            "faux",
        }:
            return False

    return None


def _product_is_new(
    product: Product,
) -> bool:
    """
    Determine whether a product is new.

    Prefer an explicit is_new attribute when available. Otherwise fall back
    to the textual condition field.
    """
    explicit_value = getattr(
        product,
        "is_new",
        None,
    )

    parsed = _parse_optional_bool(
        explicit_value
    )

    if parsed is not None:
        return parsed

    condition = _normalise_text(
        getattr(product, "condition", None)
    )

    return condition in {
        "neuf",
        "new",
        "nouveau",
        "nouvelle",
        "jamais utilise",
        "jamais utilisee",
    }


def _product_is_used(
    product: Product,
) -> bool:
    """
    Determine whether a product is used.

    Prefer an explicit is_used attribute when available. Otherwise fall back
    to the textual condition field.
    """
    explicit_value = getattr(
        product,
        "is_used",
        None,
    )

    parsed = _parse_optional_bool(
        explicit_value
    )

    if parsed is not None:
        return parsed

    condition = _normalise_text(
        getattr(product, "condition", None)
    )

    return condition in {
        "occasion",
        "used",
        "seconde main",
        "second hand",
    }
