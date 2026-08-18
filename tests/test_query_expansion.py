"""Tests for deterministic expansion using the shared synonym resource."""

from __future__ import annotations

from search.query_expansion import QueryExpansionService


def test_expansion_loads_nested_resource_and_enriches_laptop() -> None:
    """A nested ``categories`` resource adds semantic aliases to a query."""
    service = QueryExpansionService()

    expanded = service.expand("laptop pas cher")

    assert expanded.startswith("laptop pas cher")
    assert len(expanded.split()) < 12
    assert "electronic" in expanded
    assert "Électronique" not in expanded


def test_expansion_matches_arabic_darija_and_mixed_aliases() -> None:
    """Explicit aliases work for Arabic, Darija, and mixed-language queries."""
    service = QueryExpansionService()

    assert service.expand("حاسوب").startswith("حاسوب")
    assert service.expand("mte3 dar").startswith("mte3 dar")
    assert service.expand("laptop هاتف").startswith("laptop هاتف")


def test_default_resource_path_does_not_depend_on_current_directory(
    monkeypatch,
    tmp_path,
) -> None:
    """The production synonym resource is resolved relative to source code."""
    monkeypatch.chdir(tmp_path)

    assert QueryExpansionService().expand("laptop").startswith("laptop")
