"""Tests for product-focused semantic-query construction."""

from __future__ import annotations

from models.query_analysis import QueryIntent, SupportedLanguage
from models.search_query import StructuredSearchQuery
from search.query_expansion import QueryExpansionService
from search.semantic_query_builder import SemanticQueryBuilder


def _query(original: str, **values: object) -> StructuredSearchQuery:
    defaults: dict[str, object] = {
        "original_query": original,
        "semantic_query": original,
        "language": SupportedLanguage.UNKNOWN,
        "intent": QueryIntent.PRODUCT_SEARCH,
    }
    defaults.update(values)
    return StructuredSearchQuery(**defaults)  # type: ignore[arg-type]


def test_arabic_footwear_keeps_product_concept_not_audience() -> None:
    result = SemanticQueryBuilder(QueryExpansionService()).build(
        _query(
            "بغيت سبرديلة ديال الرجال",
            category="Hommes",
            category_hint="الرجال",
            product_type="chaussures",
            product_type_hint="سبرديلة",
        )
    )

    assert result.text == "سبرديلة chaussures shoes sberdila"
    assert "الرجال" not in result.text
    assert result.removed_constraints == ("الرجال",)


def test_french_footwear_does_not_embed_resolved_audience() -> None:
    result = SemanticQueryBuilder(QueryExpansionService()).build(
        _query(
            "chaussures homme",
            category="Hommes",
            product_type="chaussures",
        )
    )

    assert result.text.startswith("chaussures")
    assert "homme" not in result.text


def test_unknown_product_and_new_brand_are_preserved() -> None:
    result = SemanticQueryBuilder(QueryExpansionService()).build(
        _query(
            "bghit Zyrqophone X99 dial rjal",
            category="Hommes",
            brand="Zyrqophone",
        )
    )

    assert "Zyrqophone X99" in result.text
    assert result.text == "bghit Zyrqophone X99 dial rjal"


def test_category_alias_that_is_also_product_concept_is_not_deleted() -> None:
    result = SemanticQueryBuilder(QueryExpansionService()).build(
        _query("gaming laptop", category="Électronique")
    )

    assert result.text == "gaming laptop"


def test_exact_reliable_attribute_is_removed_without_losing_product() -> None:
    result = SemanticQueryBuilder(QueryExpansionService()).build(
        _query("bghit parfum noir", color="noir")
    )

    assert result.text == "bghit parfum"
    assert result.removed_constraints == ("noir",)
