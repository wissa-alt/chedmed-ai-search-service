"""

Multilingual semantic query understanding.

This module analyses a user's search request before semantic search.

Supported languages:

- French
- English
- Arabic
- Moroccan Darija
- Mixed-language queries

The service never answers the user.
It only analyses the request.

Category responsibility:

- The LLM extracts raw category and product-type hints.
- This service preserves those hints without resolving business values.
- Deterministic resolvers downstream map hints to catalogue vocabulary.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from models.query_analysis import (
    QueryAnalysis,
    QueryIntent,
    SearchFilters,
    SupportedLanguage,
)

LOGGER = logging.getLogger(__name__)


class QueryUnderstandingError(RuntimeError):
    """Raised when the query cannot be analysed."""


class QueryUnderstandingPort(Protocol):
    """Minimal interface required from the Groq adapter."""

    def understand_query(
        self,
        *,
        model: str,
        temperature: float,
        response_format: dict[str, str],
        messages: list[dict[str, str]],
    ) -> Any:
        """Execute a structured Groq completion."""


class QueryUnderstandingService:
    """
    Understand multilingual user queries using Groq Llama.

    Responsibilities:

    1. Send the user query to the LLM.
    2. Parse the structured semantic response.
    3. Preserve raw category and product-type hints returned by the LLM.

    This service does NOT:
        - perform embeddings;
        - perform FAISS search;
        - retrieve products;
        - apply product filters.
    """

    def __init__(
        self,
        groq_client: QueryUnderstandingPort,
        model: str,
        category_normalizer: object | None = None,
    ) -> None:
        """
        Initialize the query understanding service.

        Args:
            groq_client:
                Adapter used to communicate with Groq.

            model:
                Groq model name.

            category_normalizer:
                Optional category normalizer.

                Dependency injection is used here so tests can provide
                a controlled normalizer.
        """

        self._client = groq_client
        self._model = model

        # Retained only for constructor compatibility. Business resolution is
        # deliberately performed downstream by SearchService.
        self._legacy_category_normalizer = category_normalizer

    def understand(
        self,
        query: str,
    ) -> QueryAnalysis:
        """
        Analyse a user query and return a structured QueryAnalysis.

        Raw semantic hints are always preserved for downstream resolution.
        """

        if not isinstance(query, str):
            raise QueryUnderstandingError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise QueryUnderstandingError(
                "Query cannot be empty."
            )

        prompt = self._build_prompt(query)

        try:
            # ---------------------------------------------------------
            # 1. Call LLM
            # ---------------------------------------------------------

            response = self._client.understand_query(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert multilingual semantic "
                            "search engine."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            content = response.choices[0].message.content

            if not isinstance(content, str) or not content.strip():
                raise QueryUnderstandingError(
                    "Groq returned an empty semantic response."
                )

            content = content.strip()

            LOGGER.info(
                "Groq semantic response:\n%s",
                content,
            )

            # ---------------------------------------------------------
            # 2. Parse JSON
            # ---------------------------------------------------------

            data = json.loads(content)

            if not isinstance(data, dict):
                raise QueryUnderstandingError(
                    "Groq semantic response must be a JSON object."
                )

            # ---------------------------------------------------------
            # 3. Convert to QueryAnalysis
            # ---------------------------------------------------------

            return self._parse_response(
                query=query,
                data=data,
            )

        except QueryUnderstandingError:
            LOGGER.exception(
                "Semantic understanding failed."
            )
            return QueryAnalysis.empty(query)

        except json.JSONDecodeError:
            LOGGER.exception(
                "Groq returned invalid JSON for semantic understanding."
            )
            return QueryAnalysis.empty(query)

        except Exception:
            LOGGER.exception(
                "Semantic understanding failed."
            )
            return QueryAnalysis.empty(query)

    @staticmethod
    def _build_prompt(
        query: str,
    ) -> str:
        """
        Build the structured multilingual analysis prompt.

        IMPORTANT:

        The LLM must return the semantic category exactly as understood
        from the user query.

        It must NOT convert it to an official ChedMed category.

        Category normalization is handled by CategoryNormalizer.
        """

        return f"""
You are a multilingual semantic search engine.

Understand the following languages:

- French
- English
- Arabic
- Moroccan Darija
- Mixed language

Your task is NOT to answer the user.

Return ONLY a JSON object.

Schema:

{{
    "language": "fr|en|ar|darija|mixed|unknown",
    "intent": "product_search|product_comparison|product_recommendation|unknown",
    "expanded_query": "...",
    "keywords": [
        "...",
        "..."
    ],
    "filters": {{
        "brand": null,
        "category": null,
        "product_type": null,
        "city": null,
        "color": null,
        "condition": null,
        "min_price": null,
        "max_price": null,
        "currency": "MAD",
        "is_new": null,
        "is_used": null,
        "is_sold": null
    }},
    "confidence": 0.95,
    "reasoning": "short explanation"
}}

Rules:

- Correct spelling mistakes.
- Expand abbreviations.
- Understand Darija.
- Understand Arabic.
- Understand French.
- Understand English.
- Understand mixed queries.
- Preserve intent.
- Never invent products.
- Never invent brands.
- Never answer the user.
- Detect the user's language.
- Detect if multiple languages are mixed.
- Detect product attributes.
- Detect brands.
- Detect colors.
- Detect cities.
- Detect categories.
- Detect price constraints.
- Detect product condition.
- Detect comparison requests.
- Detect recommendation requests.

IMPORTANT CATEGORY / PRODUCT TYPE RULE:

- "category" must contain the broad semantic category or audience
  explicitly requested by the user.

- "product_type" must contain the specific type of product requested.

Examples:

"chaussures de sport pour hommes"
    category = "hommes"
    product_type = "chaussures de sport"

"sneakers noir pour homme"
    category = "hommes"
    product_type = "sneakers"

"ordinateur portable Dell"
    category = null
    product_type = "ordinateur portable"
    brand = "Dell"

"robe noire pour femme"
    category = "femmes"
    product_type = "robe"
    color = "noir"

"téléphone Samsung"
    category = null
    product_type = "téléphone"
    brand = "Samsung"

- Do NOT put the entire phrase inside "category".
- Do NOT use an official ChedMed category as the semantic category.
- Do NOT convert "sneakers" into "Électronique", "Maison", "Hommes", etc.
- Preserve the natural semantic meaning.

 EXPANDED QUERY RULES:

- Rewrite the query to maximize semantic retrieval.
- Add useful synonyms in the detected languages.
- Never remove important keywords.
- Keep the query in natural language.
- Do NOT replace the semantic category by an official ChedMed
  category.
- Do NOT force labels such as "Électronique" into the expanded query.
- Never invent brands.
- Never invent products.
- Never invent filters.

Return valid JSON only.

User query:

{query}
"""

    def _parse_response(
        self,
        query: str,
        data: dict[str, Any],
    ) -> QueryAnalysis:
        """
        Convert Groq JSON into QueryAnalysis.

        Provider fields are validated here; business resolution is downstream.
        """

        confidence = 0.0

        try:
            confidence = float(
                data.get("confidence", 0)
            )
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        # -------------------------------------------------------------
        # 1. Parse filters
        # -------------------------------------------------------------

        filters_data = data.get(
            "filters",
            {},
        )

        if not isinstance(filters_data, dict):
            filters_data = {}

        # ``category`` is the stable provider JSON field.  ``category_raw``
        # is accepted for forward-compatible providers, but both values mean
        # the same thing: a semantic category that has not yet been mapped to
        # an official ChedMed category.
        category_raw = _clean_optional_string(
            filters_data.get("category_raw", filters_data.get("category"))
        )

        LOGGER.info(
            "Catégorie brute extraite par le LLM : %r",
            category_raw,
        )

        filters = self._parse_filters(
            filters_data,
            category_raw=category_raw,
            category_normalized=None,
        )

        # -------------------------------------------------------------
        # 5. Build QueryAnalysis
        # -------------------------------------------------------------

        return QueryAnalysis(
            original_query=query,

            language=self._parse_language(
                data.get("language")
            ),

            intent=self._parse_intent(
                data.get("intent")
            ),

            # IMPORTANT:
            # expanded_query remains natural language.
            # category_normalized is NOT injected into it.
            expanded_query=str(
                data.get("expanded_query") or query
            ).strip(),

            keywords=self._safe_list(
                data.get("keywords")
            ),

            filters=filters,

            confidence=confidence,

            reasoning=(
                str(data.get("reasoning"))
                if data.get("reasoning") is not None
                else None
            ),
        )

    @staticmethod
    def _parse_language(
        value: str | None,
    ) -> SupportedLanguage:
        """Convert the provider language value to the domain enum."""

        try:
            return SupportedLanguage(value)
        except (ValueError, TypeError):
            return SupportedLanguage.UNKNOWN

    @staticmethod
    def _parse_intent(
        value: str | None,
    ) -> QueryIntent:
        """Convert the provider intent value to the domain enum."""

        try:
            return QueryIntent(value)
        except (ValueError, TypeError):
            return QueryIntent.UNKNOWN

    @staticmethod
    def _safe_list(
        value: Any,
    ) -> list[str]:
        """Return only non-empty string-compatible keyword values."""

        if not isinstance(value, list):
            return []

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    @staticmethod
    def _parse_filters(
        data: Any,
        *,
        category_raw: str | None = None,
        category_normalized: str | None = None,
    ) -> SearchFilters:
        """
        Convert provider filters into validated domain filters.

        Category values are stored here as the canonical category-filter
        state. ``category_raw`` always preserves the LLM's semantic value;
        ``category_normalized`` is populated exactly once immediately after
        the provider response is parsed.
        """

        if not isinstance(data, dict):
            return SearchFilters(
                currency="MAD",
                category_raw=category_raw,
                category_normalized=category_normalized,
            )
        product_type_raw = _clean_optional_string(
            data.get("product_type_raw", data.get("product_type"))
        )

        brand = _clean_optional_string(
            data.get("brand")
        )

        city = _clean_optional_string(
            data.get("city")
        )

        color = _clean_optional_string(
            data.get("color")
        )

        condition = _clean_optional_string(
            data.get("condition")
        )

        min_price = _parse_price(
            data.get("min_price")
        )

        max_price = _parse_price(
            data.get("max_price")
        )

        currency = _normalize_currency(
            data.get("currency")
        )

        return SearchFilters(
            category_raw=category_raw,
            category_normalized=category_normalized,
            product_type_raw=product_type_raw,
            product_type_normalized=None,
            brand=brand,

            city=city,
            color=color,
            condition=condition,
            min_price=min_price,
            max_price=max_price,
            currency=currency,
            is_new=_parse_optional_bool(
                data.get("is_new")
            ),
            is_used=_parse_optional_bool(
                data.get("is_used")
            ),
            is_sold=_parse_optional_bool(
                data.get("is_sold")
            ),
        )


def _clean_optional_string(
    value: Any,
) -> str | None:
    """Return a cleaned optional string or None."""

    if value is None:
        return None

    if not isinstance(value, str):
        return None

    value = value.strip()

    return value or None


def _parse_price(
    value: Any,
) -> float | None:
    """
    Convert a provider price value to a finite float.

    Textual semantic expressions are deliberately rejected rather than
    converted into an arbitrary numeric value.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        numeric_value = float(value)

        if numeric_value >= 0:
            return numeric_value

        return None

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        try:
            numeric_value = float(
                value.replace(",", ".")
            )
        except ValueError:
            return None

        if numeric_value >= 0:
            return numeric_value

    return None


def _normalize_currency(
    value: Any,
) -> str:
    """
    Normalize supported currency codes.

    MAD is the default ChedMed catalogue currency.
    """

    if not isinstance(value, str):
        return "MAD"

    normalized = value.strip().upper()

    if normalized in {
        "MAD",
        "DH",
        "DHS",
        "DIRHAM",
        "DIRHAMS",
    }:
        return "MAD"

    if normalized in {
        "EUR",
        "EURO",
        "EUROS",
        "€",
    }:
        return "EUR"

    if normalized in {
        "USD",
        "DOLLAR",
        "DOLLARS",
        "$",
    }:
        return "USD"

    return "MAD"


def _parse_optional_bool(
    value: Any,
) -> bool | None:
    """Convert a provider boolean value safely."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "yes",
            "oui",
            "1",
        }:
            return True

        if normalized in {
            "false",
            "no",
            "non",
            "0",
        }:
            return False

    return None
