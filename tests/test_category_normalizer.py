
"""
Tests for the ChedMed category normalization layer.

The category normalizer is responsible for converting semantic category
expressions into the canonical categories used by the ChedMed catalogue.

Current catalogue categories observed in ChedMed:

- Électronique
- Maison
- Hommes
- Femmes
- Enfants
- Divertissement
- Fabriqué au Maroc

These tests intentionally do not call:
- Groq
- FAISS
- SentenceTransformers
- ChedMed API

The tests must remain deterministic and isolated.
"""

from __future__ import annotations

import pytest

from search.category_normalizer import (
    CategoryNormalizer,
    CategoryNormalizerError,
)


@pytest.fixture
def normalizer() -> CategoryNormalizer:
    """Return a fresh category normalizer."""
    return CategoryNormalizer()


# ---------------------------------------------------------------------------
# Basic normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Électronique", "Électronique"),
        ("electronique", "Électronique"),
        ("électronique", "Électronique"),
        (" ELECTRONIQUE ", "Électronique"),
        ("Maison", "Maison"),
        ("maison", "Maison"),
        ("HOMMES", "Hommes"),
        ("hommes", "Hommes"),
        ("Femmes", "Femmes"),
        ("femmes", "Femmes"),
        ("Enfants", "Enfants"),
        ("enfants", "Enfants"),
        ("Divertissement", "Divertissement"),
        ("divertissement", "Divertissement"),
        ("Fabriqué au Maroc", "Fabriqué au Maroc"),
        ("fabrique au maroc", "Fabriqué au Maroc"),
    ],
)
def test_normalize_known_categories(
    normalizer: CategoryNormalizer,
    value: str,
    expected: str,
) -> None:
    """Known catalogue categories are converted to canonical values."""

    assert normalizer.normalize(value) == expected


# ---------------------------------------------------------------------------
# Semantic aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Informatique belongs to the current ChedMed
        # Électronique catalogue category.
        ("informatique", "Électronique"),
        ("Informatique", "Électronique"),
        ("informatique et technologie", "Électronique"),
        ("ordinateur", "Électronique"),
        ("ordinateurs", "Électronique"),
        ("laptop", "Électronique"),
        ("laptops", "Électronique"),
        ("LAPTOP", "Électronique"),
        ("Laptop", "Électronique"),
        ("pc", "Électronique"),
        ("pc portable", "Électronique"),
        ("ordinateur portable", "Électronique"),
        ("ordinateur laptop", "Électronique"),
        ("smartphone", "Électronique"),
        ("smartphones", "Électronique"),
        ("électroménager", "Maison"),
        ("maisons", "Maison"),
        ("meubles", "Maison"),
        ("mobilier", "Maison"),
        ("homme", "Hommes"),
        ("hommes", "Hommes"),
        ("mode homme", "Hommes"),
        ("vêtements homme", "Hommes"),
        ("femme", "Femmes"),
        ("femmes", "Femmes"),
        ("mode femme", "Femmes"),
        ("vêtements femme", "Femmes"),
        ("enfant", "Enfants"),
        ("enfants", "Enfants"),
        ("jouets", "Enfants"),
        ("jeux pour enfants", "Enfants"),
        ("jeu", "Divertissement"),
        ("jeux", "Divertissement"),
        ("made in morocco", "Fabriqué au Maroc"),
        ("حاسوب", "Électronique"),
        ("بيسي", "Électronique"),
        ("mte3 dar", "Maison"),
    ],
)
def test_normalize_category_aliases(
    normalizer: CategoryNormalizer,
    value: str,
    expected: str,
) -> None:
    """Semantic category aliases resolve to catalogue categories."""

    assert normalizer.normalize(value) == expected


# ---------------------------------------------------------------------------
# Accent and case handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "electronique",
        "ÉLECTRONIQUE",
        "éLeCtRoNiQuE",
        "  électronique  ",
        "FABRIQUE AU MAROC",
        "fabriqué au maroc",
    ],
)
def test_normalize_is_case_and_accent_tolerant(
    normalizer: CategoryNormalizer,
    value: str,
) -> None:
    """Normalization is tolerant of case and common accent variations."""

    result = normalizer.normalize(value)

    assert isinstance(result, str)
    assert result


# ---------------------------------------------------------------------------
# Unknown categories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "automobile",
        "voiture",
        "pharmacie",
        "informatique quantique",
        "catégorie inexistante",
        "something completely unknown",
    ],
)
def test_unknown_category_returns_none(
    normalizer: CategoryNormalizer,
    value: str,
) -> None:
    """Unknown categories must not be invented."""

    assert normalizer.normalize(value) is None


def test_unknown_category_logs_warning(
    normalizer: CategoryNormalizer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown categories remain unresolved and are observable in logs."""
    with caplog.at_level("WARNING"):
        assert normalizer.normalize("astronautes") is None

    assert "Unknown category: astronautes" in caplog.text


# ---------------------------------------------------------------------------
# Empty / invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
    ],
)
def test_empty_or_none_category_returns_none(
    normalizer: CategoryNormalizer,
    value: str | None,
) -> None:
    """Empty category values are treated as unresolved."""

    assert normalizer.normalize(value) is None


@pytest.mark.parametrize(
    "value",
    [
        123,
        12.5,
        [],
        {},
        object(),
    ],
)
def test_invalid_category_type_raises_error(
    normalizer: CategoryNormalizer,
    value: object,
) -> None:
    """Unsupported input types must be rejected safely."""

    with pytest.raises(CategoryNormalizerError):
        normalizer.normalize(value)


# ---------------------------------------------------------------------------
# Catalogue validation
# ---------------------------------------------------------------------------


def test_canonical_categories_are_available(
    normalizer: CategoryNormalizer,
) -> None:
    """The normalizer exposes the categories used by ChedMed."""

    categories = normalizer.categories()

    assert "Électronique" in categories
    assert "Maison" in categories
    assert "Hommes" in categories
    assert "Femmes" in categories
    assert "Enfants" in categories
    assert "Divertissement" in categories
    assert "Fabriqué au Maroc" in categories


def test_categories_are_unique(
    normalizer: CategoryNormalizer,
) -> None:
    """Canonical categories must not contain duplicates."""

    categories = normalizer.categories()

    assert len(categories) == len(set(categories))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "product_category", "expected"),
    [
        ("informatique", "Électronique", True),
        ("ordinateur portable", "Électronique", True),
        ("laptop", "Électronique", True),
        ("électronique", "Électronique", True),
        ("maison", "Maison", True),
        ("hommes", "Hommes", True),
        ("femmes", "Femmes", True),
        ("enfants", "Enfants", True),
        ("divertissement", "Divertissement", True),
        ("informatique", "Maison", False),
        ("informatique", "Hommes", False),
        ("femmes", "Hommes", False),
        ("maison", "Électronique", False),
    ],
)
def test_matches_category(
    normalizer: CategoryNormalizer,
    requested: str,
    product_category: str,
    expected: bool,
) -> None:
    """Semantic requests are correctly compared to catalogue categories."""

    assert (
        normalizer.matches(
            requested,
            product_category,
        )
        is expected
    )


# ---------------------------------------------------------------------------
# Real ChedMed search case
# ---------------------------------------------------------------------------


def test_ordinateur_portable_matches_electronique(
    normalizer: CategoryNormalizer,
) -> None:
    """
    A query for 'ordinateur portable' must match an Électronique product.

    This is the regression test for the current search problem where Groq
    correctly produces:

        category = 'informatique'

    while ChedMed actually stores:

        category = 'Électronique'
    """

    requested_category = "informatique"
    product_category = "Électronique"

    assert normalizer.matches(
        requested_category,
        product_category,
    )


def test_ordinateur_portable_does_not_match_maison(
    normalizer: CategoryNormalizer,
) -> None:
    """An informatique query must not match a Maison product."""

    assert not normalizer.matches(
        "informatique",
        "Maison",
    )


# ---------------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------------


def test_normalize_does_not_modify_input(
    normalizer: CategoryNormalizer,
) -> None:
    """Normalization must not mutate the caller's string."""

    value = "  informatique  "

    normalizer.normalize(value)

    assert value == "  informatique  "
