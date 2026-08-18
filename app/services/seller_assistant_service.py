"""Seller assistance based on existing catalogue search and robust price statistics."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from models.product import Product
from models.query_analysis import QueryIntent, SupportedLanguage
from models.search_query import SearchSource, StructuredSearchQuery
from search.product_families import ProductFamilies
from search.product_filter import ProductFilterService
from search.search_service import SearchResult, SearchResultItem, SearchService


class SellerAssistantValidationError(ValueError):
    """Raised when seller input cannot be safely interpreted."""


class SellerAssistantUnavailableError(RuntimeError):
    """Raised when the shared search pipeline is unavailable."""


class SellerDescriptionPort(Protocol):
    def suggest_seller_description(self, fields: Mapping[str, str | None]) -> str:
        """Return a factual description based only on supplied seller fields."""


@dataclass(frozen=True, slots=True)
class SellerProductInput:
    title: str
    seller_price: Decimal
    currency: str
    description: str = ""
    category: str | None = None
    brand: str | None = None
    color: str | None = None
    condition: str | None = None
    product_id: str | None = None
    image_observations: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SellerProductInput":
        if not isinstance(payload, Mapping):
            raise SellerAssistantValidationError("Le corps doit être un objet JSON.")
        title = _required_text(payload.get("title"), "title")
        price = _positive_decimal(payload.get("sellerPrice"), "sellerPrice")
        currency = _required_text(payload.get("currency"), "currency").upper()
        if not re.fullmatch(r"[A-Z]{3}", currency):
            raise SellerAssistantValidationError(
                "currency doit être un code de devise ISO à trois lettres."
            )
        return cls(
            title=title,
            description=_optional_text(payload.get("description")) or "",
            category=_optional_text(payload.get("category")),
            brand=_optional_text(payload.get("brand")),
            color=_optional_text(payload.get("color")),
            condition=_optional_text(payload.get("condition")),
            seller_price=price,
            currency=currency,
            product_id=_optional_text(payload.get("productId")),
            image_observations=_optional_text(payload.get("imageObservations")),
        )


@dataclass(frozen=True, slots=True)
class ComparableProduct:
    product: Product
    match_type: str
    similarity_score: float


@dataclass(frozen=True, slots=True)
class RecommendedRange:
    minimum: Decimal
    maximum: Decimal


@dataclass(frozen=True, slots=True)
class SellerAssistantResult:
    suggested_description: str
    description_generated: bool
    description_quality: str
    seller_price: Decimal
    currency: str
    estimated_price: Decimal | None
    recommended_range: RecommendedRange | None
    price_assessment: str
    message: str
    confidence: str
    comparables_count: int
    comparables: tuple[ComparableProduct, ...]


@dataclass(frozen=True, slots=True)
class PriceEstimateResult:
    suggested_price: Decimal | None
    mean_price: Decimal | None
    minimum: Decimal | None
    maximum: Decimal | None
    comparables: tuple[ComparableProduct, ...]
    candidate_products_count: int = 0
    total_catalog_products: int = 0


@dataclass(frozen=True, slots=True)
class MarketStats:
    mean: Decimal
    median: Decimal
    p25: Decimal
    p75: Decimal
    minimum: Decimal
    maximum: Decimal


@dataclass(frozen=True, slots=True)
class PriceCheckResult:
    alert: str
    message: str
    seller_price: Decimal
    stats: MarketStats | None
    comparables_count: int
    comparables: tuple[ComparableProduct, ...] = ()
    candidate_products_count: int = 0
    total_catalog_products: int = 0


@dataclass(frozen=True, slots=True)
class ComparableSelection:
    """One shared Seller view over a full-catalogue SearchService result."""

    comparables: tuple[ComparableProduct, ...]
    candidate_products_count: int
    total_catalog_products: int


class SellerAssistantService:
    """Consume SearchService results and calculate an explainable market estimate."""

    _SEARCH_LIMIT = 20
    _PUBLIC_COMPARABLES_LIMIT = 5
    _USABLE_STATUSES = frozenset({"ACCEPTED", "ACTIVE"})

    def __init__(
        self,
        search_service: SearchService,
        description_provider: SellerDescriptionPort | None = None,
        *,
        product_families: ProductFamilies | None = None,
    ) -> None:
        self._search = search_service
        self._description_provider = description_provider
        self._families = product_families or ProductFamilies()
        self._brand_filter = ProductFilterService()

    def assist(self, seller: SellerProductInput) -> SellerAssistantResult:
        query = self.build_comparable_query(seller)
        try:
            search_result = self._search.search(
                query, self._SEARCH_LIMIT, source=SearchSource.TEXT
            )
        except Exception as exc:
            raise SellerAssistantUnavailableError(
                "La recherche de produits comparables est indisponible."
            ) from exc

        comparables = self._select_comparables(seller, search_result)
        retained, estimated, price_range = _estimate_prices(comparables)
        confidence = _confidence(seller, retained, estimated, price_range)
        assessment = _assessment(seller.seller_price, price_range)
        description, generated = self._description(seller)
        return SellerAssistantResult(
            suggested_description=description,
            description_generated=generated,
            description_quality=_description_quality(seller),
            seller_price=seller.seller_price,
            currency=seller.currency,
            estimated_price=estimated,
            recommended_range=price_range,
            price_assessment=assessment,
            message=_ASSESSMENT_MESSAGES[assessment],
            confidence=confidence,
            comparables_count=len(retained),
            comparables=tuple(retained[: self._PUBLIC_COMPARABLES_LIMIT]),
        )

    def suggest_description(
        self,
        product_name: str,
        category: str,
        keywords: str | None = None,
        language: str = "fr",
        image_analysis: str | None = None,
    ) -> tuple[str, bool]:
        """Generate one grounded commercial description with deterministic fallback."""
        product_name = _required_text(product_name, "product_name")
        category = _required_text(category, "category")
        language = _optional_text(language) or "fr"
        fields = {
            "product_name": product_name,
            "category": category,
            "keywords": _optional_text(keywords),
            "language": language,
            "visibleImageObservations": _optional_text(image_analysis),
        }
        fallback = f"{product_name} proposé à la vente dans la catégorie {category}."
        if keywords:
            fallback = f"{fallback[:-1]}. {' '.join(keywords.split())}."
        if self._description_provider is None:
            return fallback, False
        try:
            description = self._description_provider.suggest_seller_description(fields)
        except Exception:
            return fallback, False
        return (description.strip(), True) if description.strip() else (fallback, False)

    def estimate_price(
        self,
        description: str,
        category: str | None = None,
        *,
        currency: str = "MAD",
    ) -> PriceEstimateResult:
        """Return catalogue statistics and a robust median-based suggestion."""
        selection = self._comparables_for_description(description, category, currency)
        comparables = list(selection.comparables)
        retained, suggested, _ = _estimate_prices(comparables)
        prices = [item.product.price for item in comparables]
        if suggested is None and prices:
            suggested = _quantile(sorted(prices), Decimal("0.5"))
        mean = sum(prices, Decimal(0)) / Decimal(len(prices)) if prices else None
        return PriceEstimateResult(
            suggested_price=suggested,
            mean_price=mean,
            minimum=min(prices) if prices else None,
            maximum=max(prices) if prices else None,
            comparables=tuple(comparables),
            candidate_products_count=selection.candidate_products_count,
            total_catalog_products=selection.total_catalog_products,
        )

    def check_price(
        self,
        description: str,
        seller_price: object,
        category: str | None = None,
        *,
        currency: str = "MAD",
    ) -> PriceCheckResult:
        """Classify a seller price from observed quartiles, never an LLM opinion."""
        price = _positive_decimal(seller_price, "seller_price")
        selection = self._comparables_for_description(description, category, currency)
        prices = sorted(item.product.price for item in selection.comparables)
        if not prices:
            return PriceCheckResult(
                "insufficient_data",
                "Nous n’avons pas assez de produits comparables pour contrôler ce prix.",
                price,
                None,
                0,
                (),
                selection.candidate_products_count,
                selection.total_catalog_products,
            )
        stats = MarketStats(
            mean=sum(prices, Decimal(0)) / Decimal(len(prices)),
            median=_quantile(prices, Decimal("0.5")),
            p25=_quantile(prices, Decimal("0.25")),
            p75=_quantile(prices, Decimal("0.75")),
            minimum=prices[0],
            maximum=prices[-1],
        )
        alert = _check_price_alert(price, stats)
        messages = {
            "too_low": "Prix très bas par rapport aux produits comparables du marché.",
            "low": "Prix légèrement inférieur à la fourchette centrale du marché.",
            "fair": "Prix cohérent avec les produits comparables du marché.",
            "high": "Prix légèrement supérieur à la fourchette centrale du marché.",
            "too_high": "Prix très élevé par rapport aux produits comparables du marché.",
        }
        return PriceCheckResult(
            alert, messages[alert], price, stats, len(prices),
            selection.comparables,
            selection.candidate_products_count,
            selection.total_catalog_products,
        )

    def _comparables_for_description(
        self, description: str, category: str | None, currency: str
    ) -> ComparableSelection:
        description = _required_text(description, "description")
        currency = _required_text(currency, "currency").upper()
        seller = SellerProductInput(
            title=description,
            description="",
            category=_optional_text(category),
            seller_price=Decimal(1),
            currency=currency,
        )
        try:
            result = self._search.search(
                description, None, source=SearchSource.TEXT, include_all=True
            )
            market_query = self._market_query(result, description)
            if _normalize(market_query) != _normalize(description):
                result = self._search.search(
                    market_query, None, source=SearchSource.TEXT, include_all=True
                )
        except Exception as exc:
            raise SellerAssistantUnavailableError(
                "La recherche de produits comparables est indisponible."
            ) from exc
        comparables = self._select_comparables(
            seller, result, include_historical=True
        )
        return ComparableSelection(
            tuple(comparables),
            result.candidate_products_count or result.faiss_candidates_count,
            result.total_catalog_products,
        )

    @staticmethod
    def _market_query(result: SearchResult, original: str) -> str:
        """Remove model/style details only when understanding exposed safe hints.

        This broadens a seller market cohort through the same SearchService;
        it does not inspect catalogue products or invent a product concept.
        """
        query = result.structured_query
        if query is None:
            return original
        product_type = query.product_type or query.product_type_hint
        if not product_type:
            return original
        parts: list[str] = []
        for value in (query.brand, product_type, query.category_hint):
            if value and _normalize(value) not in _normalize(" ".join(parts)):
                parts.append(value)
        return " ".join(parts) or original

    def build_comparable_query(self, seller: SellerProductInput) -> str:
        """Build concise semantic text without seller price or duplicate attributes."""
        values = (
            seller.title,
            seller.description[:300] or None,
            seller.brand,
            seller.category,
            seller.condition,
            seller.color,
            seller.image_observations,
        )
        parts: list[str] = []
        normalized = ""
        for value in values:
            if not value:
                continue
            candidate = " ".join(value.split())
            key = _normalize(candidate)
            if not key or key in normalized:
                continue
            parts.append(candidate)
            normalized = _normalize(" ".join(parts))
        return " ".join(parts)

    def _select_comparables(
        self,
        seller: SellerProductInput,
        result: SearchResult,
        *,
        include_historical: bool = False,
    ) -> list[ComparableProduct]:
        requested_family = self._requested_family(seller, result)
        structured = result.structured_query
        explicit_category = (
            structured.category
            if structured is not None and structured.category_hint
            else None
        )
        main_categories = {
            _normalize(item.product.category)
            for item in result.items
            if item.match_type in {"exact", "relevant", "similar"}
            and item.product.category
        }
        selected: list[ComparableProduct] = []
        for item in result.items:
            product = item.product
            if seller.product_id and product.id == seller.product_id:
                continue
            if not include_historical and (
                product.is_sold or product.status.upper() not in self._USABLE_STATUSES
            ):
                continue
            if product.currency.upper() != seller.currency:
                continue
            price = _catalogue_price(product.price)
            if price is None:
                continue
            if not include_historical and seller.brand and not self._brand_filter.matches(
                product, _brand_filters(seller.brand)
            ):
                continue
            product = replace(product, price=price)
            product_family = self._families.for_product(product)
            if item.match_type == "unrelated":
                continue
            if explicit_category and _normalize(product.category) != _normalize(explicit_category):
                continue
            if requested_family:
                same_family = self._families.matches(product, requested_family)
                broad_family = self._families.broad_matches(product, requested_family)
                if item.match_type == "broad_similar":
                    if not broad_family:
                        continue
                    title_evidence = self._families.broad_matches(
                        replace(product, description="", brand=None), requested_family
                    )
                    if not title_evidence and _normalize(product.category) not in main_categories:
                        continue
                elif item.match_type == "similar":
                    if not same_family:
                        continue
                elif item.match_type in {"exact", "relevant"}:
                    # A model name such as Latitude may be strong relevance
                    # evidence even when the compact family vocabulary cannot
                    # resolve the catalogue title itself.
                    if not same_family and product_family is not None:
                        continue
                else:
                    continue
            if requested_family is None and (
                item.match_type != "relevant"
                or item.relevance_reason != "focal_lexical_evidence"
                or not _meaningful_lexical_evidence(item)
            ):
                continue
            selected.append(
                ComparableProduct(product, item.match_type, item.score)
            )
        return sorted(
            selected,
            key=lambda item: (
                {
                    "exact": 4,
                    "relevant": 3,
                    "similar": 2,
                    "broad_similar": 1,
                }.get(item.match_type, 0),
                _same_condition(seller.condition, item.product.condition),
                item.similarity_score,
            ),
            reverse=True,
        )

    def _requested_family(
        self, seller: SellerProductInput, result: SearchResult
    ) -> str | None:
        if result.structured_query is not None:
            family = self._families.for_query(result.structured_query)
            if family:
                return family
        query = self.build_comparable_query(seller)
        return self._families.for_query(
            StructuredSearchQuery(
                original_query=query,
                semantic_query=query,
                language=SupportedLanguage.UNKNOWN,
                intent=QueryIntent.PRODUCT_SEARCH,
            )
        )

    def _description(self, seller: SellerProductInput) -> tuple[str, bool]:
        fallback = _fallback_description(seller)
        if self._description_provider is None:
            return fallback, False
        fields = {
            "title": seller.title,
            "description": seller.description or None,
            "category": seller.category,
            "brand": seller.brand,
            "color": seller.color,
            "condition": seller.condition,
            "imageObservations": seller.image_observations,
        }
        try:
            suggested = self._description_provider.suggest_seller_description(fields)
        except Exception:
            return fallback, False
        if not isinstance(suggested, str) or not suggested.strip():
            return fallback, False
        return suggested.strip(), True


def _estimate_prices(
    comparables: list[ComparableProduct],
) -> tuple[list[ComparableProduct], Decimal | None, RecommendedRange | None]:
    if not comparables:
        return [], None, None
    retained = list(comparables)
    prices = sorted(item.product.price for item in retained)
    if len(prices) >= 5:
        q1 = _quantile(prices, Decimal("0.25"))
        q3 = _quantile(prices, Decimal("0.75"))
        iqr = q3 - q1
        lower = q1 - Decimal("1.5") * iqr
        upper = q3 + Decimal("1.5") * iqr
        filtered = [item for item in retained if lower <= item.product.price <= upper]
        if len(filtered) >= 3:
            retained = filtered
            prices = sorted(item.product.price for item in retained)
    if len(prices) < 2:
        return retained, None, None
    estimated = _quantile(prices, Decimal("0.5"))
    if len(prices) == 2:
        price_range = RecommendedRange(prices[0], prices[-1])
    else:
        price_range = RecommendedRange(
            _quantile(prices, Decimal("0.25")),
            _quantile(prices, Decimal("0.75")),
        )
    return retained, estimated, price_range


def _quantile(values: list[Decimal], fraction: Decimal) -> Decimal:
    if len(values) == 1:
        return values[0]
    position = fraction * Decimal(len(values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    weight = position - Decimal(lower_index)
    return values[lower_index] + (values[upper_index] - values[lower_index]) * weight


def _assessment(price: Decimal, price_range: RecommendedRange | None) -> str:
    if price_range is None:
        return "insufficient_data"
    low, high = price_range.minimum, price_range.maximum
    spread = high - low
    if low <= price <= high:
        return "reasonable"
    if spread == 0:
        return "too_low" if price < low else "too_high"
    if price < low:
        return "low" if price >= low - spread else "too_low"
    return "high" if price <= high + spread else "too_high"


def _check_price_alert(price: Decimal, stats: MarketStats) -> str:
    """Use Tukey's inner fences around the observed central quartiles."""
    iqr = stats.p75 - stats.p25
    if iqr == 0:
        if price < stats.p25:
            return "too_low"
        if price > stats.p75:
            return "too_high"
        return "fair"
    if price < stats.p25 - iqr:
        return "too_low"
    if price < stats.p25:
        return "low"
    if price <= stats.p75:
        return "fair"
    if price <= stats.p75 + iqr:
        return "high"
    return "too_high"


def _confidence(
    seller: SellerProductInput,
    comparables: list[ComparableProduct],
    estimated: Decimal | None,
    price_range: RecommendedRange | None,
) -> str:
    count = len(comparables)
    if count == 0:
        return "none"
    if count == 1 or estimated is None:
        return "very_low"
    level = 1 if count == 2 else 2 if count <= 4 else 3
    relevant_ratio = sum(item.match_type == "relevant" for item in comparables) / count
    broad_ratio = sum(item.match_type == "broad_similar" for item in comparables) / count
    if relevant_ratio < 0.6:
        level = min(level, 1) if relevant_ratio == 0 else max(1, level - 1)
    if broad_ratio >= 0.5:
        level = 0
    if seller.condition:
        condition_ratio = sum(
            _same_condition(seller.condition, item.product.condition)
            for item in comparables
        ) / count
        if condition_ratio < 0.5:
            level = max(1, level - 1)
    if price_range and estimated and estimated > 0:
        if (price_range.maximum - price_range.minimum) / estimated > 1:
            level = max(1, level - 1)
    return ("very_low", "low", "medium", "good")[level]


_ASSESSMENT_MESSAGES = {
    "too_low": "Votre prix semble très bas par rapport aux produits similaires. Vous pourriez probablement le vendre plus cher.",
    "low": "Votre prix est légèrement inférieur à la fourchette observée.",
    "reasonable": "Votre prix est cohérent avec les produits similaires actuellement disponibles.",
    "high": "Votre prix est légèrement supérieur à la fourchette observée.",
    "too_high": "Votre prix semble nettement supérieur aux produits similaires du catalogue.",
    "insufficient_data": "Nous n’avons pas assez de produits comparables pour évaluer ce prix de manière fiable.",
}


def _fallback_description(seller: SellerProductInput) -> str:
    details = [seller.title]
    if seller.color and _normalize(seller.color) not in _normalize(" ".join(details)):
        details.append(seller.color.lower())
    if seller.condition and _normalize(seller.condition) not in _normalize(" ".join(details)):
        details.append(f"en {seller.condition.lower()}")
    sentence = " ".join(details).strip() + "."
    if seller.description:
        raw = " ".join(seller.description.split()).rstrip(".!?")
        if _normalize(raw) not in _normalize(sentence):
            sentence = f"{sentence} {raw}."
    return sentence if sentence else f"{seller.title} proposé à la vente."


def _description_quality(seller: SellerProductInput) -> str:
    supplied = sum(
        bool(value)
        for value in (
            seller.description, seller.category, seller.brand,
            seller.color, seller.condition,
        )
    )
    return "good" if supplied >= 3 else "limited"


def _brand_filters(brand: str):
    from models.query_analysis import SearchFilters
    return SearchFilters(brand=brand)


def _same_condition(requested: str | None, actual: str | None) -> bool:
    return bool(requested and actual and _normalize(requested) == _normalize(actual))


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SellerAssistantValidationError(f"{field} est obligatoire.")
    return " ".join(value.split())


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SellerAssistantValidationError("Les champs texte doivent être des chaînes.")
    return " ".join(value.split()) or None


def _positive_decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise SellerAssistantValidationError(f"{field} doit être un nombre positif.")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise SellerAssistantValidationError(f"{field} doit être un nombre positif.") from None
    if not decimal.is_finite() or decimal <= 0:
        raise SellerAssistantValidationError(f"{field} doit être un nombre positif.")
    return decimal


def _catalogue_price(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return price if price.is_finite() and price > 0 else None


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.lower(), flags=re.UNICODE))


_GENERIC_MARKETPLACE_TERMS = frozenset(
    {"produit", "product", "article", "item", "vente", "sale", "marketplace"}
)


def _meaningful_lexical_evidence(item: SearchResultItem) -> bool:
    return any(
        _normalize(term) not in _GENERIC_MARKETPLACE_TERMS
        for term in item.lexical_terms
        if len(_normalize(term)) >= 4
    )
