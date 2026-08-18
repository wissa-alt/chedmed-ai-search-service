"""Unit tests for the PostgreSQL replacement of the catalogue client."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest

from config import Settings
from database.postgres_client import (
    PostgresCatalogueClient,
    PostgresConnectionError,
    PostgresProductNotFoundError,
    PostgresQueryError,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create isolated PostgreSQL settings without environment access."""
    return Settings(
        environment="test", host="127.0.0.1", port=5000, log_level="CRITICAL",
        db_host="127.0.0.1", db_port=5432, db_name="chedmed", db_user="test",
        db_password="password", chedmed_webhook_secret="secret", groq_api_key="groq",
        project_root=tmp_path,
    )


@pytest.fixture
def connection() -> Mock:
    """Return a context-managed psycopg connection double."""
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=None)
    result = Mock()
    result.cursor.return_value = cursor
    return result


def _row() -> dict[str, object]:
    """Return one row matching the explicit PostgreSQL SELECT aliases."""
    return {
        "id": "product-1", "title": "Vélo", "description": "Vélo\nBike\nDescription",
        "category": "category-1", "brand": None, "color": None, "condition": None,
        "price": Decimal("100"), "currency": "MAD", "city": None,
        "thumbnail_url": "https://images.test/product.jpg", "image_urls": [], "status": "ACCEPTED",
        "isSold": False, "updatedAt": datetime(2026, 7, 1, tzinfo=timezone.utc),
    }


def test_get_all_products_maps_rows(settings: Settings, connection: Mock) -> None:
    """The PostgreSQL client maps a complete SQL row through Product.from_dict."""
    connection.cursor.return_value.__enter__.return_value.fetchall.return_value = [_row()]
    client = PostgresCatalogueClient(settings, connection_factory=Mock(return_value=connection))

    products = client.get_all_products()

    assert products[0].id == "product-1"
    assert products[0].image_urls == ("https://images.test/product.jpg",)


def test_get_products_page_binds_limit_and_offset(settings: Settings, connection: Mock) -> None:
    """Pagination remains compatible with the historical client interface."""
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    client = PostgresCatalogueClient(settings, connection_factory=Mock(return_value=connection))

    assert client.get_products_page(page=2, limit=25) == []
    query, params = cursor.execute.call_args.args
    assert "LIMIT %s OFFSET %s" in query
    assert params == (25, 25)


def test_get_product_raises_when_no_row_exists(settings: Settings, connection: Mock) -> None:
    """Missing products get a dedicated catalogue exception."""
    connection.cursor.return_value.__enter__.return_value.fetchall.return_value = []
    client = PostgresCatalogueClient(settings, connection_factory=Mock(return_value=connection))

    with pytest.raises(PostgresProductNotFoundError):
        client.get_product("missing")


def test_get_products_updated_after_binds_timestamp(settings: Settings, connection: Mock) -> None:
    """Incremental synchronisation keeps its original timestamp contract."""
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    client = PostgresCatalogueClient(settings, connection_factory=Mock(return_value=connection))

    client.get_products_updated_after("2026-06-01T00:00:00+00:00")
    assert cursor.execute.call_args.args[1] == ("2026-06-01T00:00:00+00:00",)


def test_all_status_incremental_query_only_excludes_deleted_rows(
    settings: Settings, connection: Mock
) -> None:
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    client = PostgresCatalogueClient(settings, connection_factory=Mock(return_value=connection))

    client.get_products_updated_after_any_status("2026-06-01T00:00:00+00:00")

    query, params = cursor.execute.call_args.args
    assert "p.deleted_at IS NULL" in query
    assert "p.is_sold IS FALSE" not in query
    assert "p.status = 'ACCEPTED'" not in query
    assert params == ("2026-06-01T00:00:00+00:00",)


def test_connection_and_query_errors_are_translated(settings: Settings) -> None:
    """Psycopg failures never leak from the catalogue adapter."""
    with pytest.raises(PostgresConnectionError):
        PostgresCatalogueClient(settings, connection_factory=Mock(side_effect=OSError("down")))

    connection = Mock()
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=None)
    cursor.execute.side_effect = RuntimeError("sql failure")
    connection.cursor.return_value = cursor
    client = PostgresCatalogueClient(settings, connection_factory=Mock(return_value=connection))

    with pytest.raises(PostgresQueryError):
        client.get_all_products()
