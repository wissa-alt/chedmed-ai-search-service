"""Unit tests for the retained ChedMed REST catalogue adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

from config import Settings
from database.api_client import ChedmedApiClient


def _settings(tmp_path: Path) -> Settings:
    """Create API-source settings without process configuration."""
    return Settings(
        environment="test", host="127.0.0.1", port=5000, log_level="CRITICAL",
        chedmed_webhook_secret="secret", catalogue_source="api",
        chedmed_api_base_url="https://chedmed.test", chedmed_api_token="token",
        groq_api_key="groq", project_root=tmp_path,
    )


def _payload() -> dict[str, object]:
    """Return a valid API Product payload."""
    return {
        "id": "product-1", "title": "Vélo", "description": "Description", "category": "Vélos",
        "brand": None, "color": None, "condition": None, "price": Decimal("100"), "currency": "MAD",
        "city": None, "imageUrls": [], "status": "ACTIVE", "isSold": False,
        "updatedAt": datetime(2026, 7, 1, tzinfo=timezone.utc).isoformat(),
    }


def test_api_client_keeps_historical_page_contract(tmp_path: Path) -> None:
    """API pagination produces typed products through the common interface."""
    session = Mock()
    session.headers = {}
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"products": [_payload()]}
    session.get.return_value = response
    client = ChedmedApiClient(_settings(tmp_path), session=session)

    products = client.get_products_page(1, 10)

    assert products[0].id == "product-1"
    session.get.assert_called_once_with(
        "https://chedmed.test/internal/ai/products", params={"page": 1, "limit": 10}, timeout=(5, 30)
    )
