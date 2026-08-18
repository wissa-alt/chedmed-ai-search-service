"""Tests for the typed Groq-to-query-analysis translation boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from llm.query_understanding import QueryUnderstandingService


class FakeGroqUnderstandingClient:
    """Return one structured Groq-like completion without network access."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.calls: list[dict[str, object]] = []

    def understand_query(self, **kwargs: object) -> object:
        """Return the injected response and retain the provider call contract."""
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self._payload)))]
        )


def test_understanding_returns_hints_without_business_resolution() -> None:
    """Understanding preserves raw hints and leaves resolution downstream."""
    client = FakeGroqUnderstandingClient(
        {
            "language": "fr",
            "intent": "product_search",
            "expanded_query": "laptop gaming",
            "keywords": ["laptop", "gaming"],
            "filters": {"category": "laptop"},
            "confidence": 0.9,
            "reasoning": "Recherche de portable de jeu.",
        }
    )
    service = QueryUnderstandingService(client, "llama-test")

    analysis = service.understand("je cherche un laptop gaming")

    assert analysis.filters.category_raw == "laptop"
    assert analysis.filters.category_normalized is None
    assert analysis.category_raw == "laptop"
    assert analysis.expanded_query == "laptop gaming"
    assert "Électronique" not in analysis.expanded_query
