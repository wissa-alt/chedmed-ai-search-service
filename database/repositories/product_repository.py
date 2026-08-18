"""PostgreSQL repository for the known ChedMed ``public.products`` table."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from database.postgres_client import PostgreSQLClient
from models.product import Product, ProductValidationError

LOGGER = logging.getLogger(__name__)


class ProductRepositoryError(RuntimeError):
    """Raised when PostgreSQL product data cannot become a valid Product."""


class ProductNotFoundError(ProductRepositoryError):
    """Raised when an active, non-sold product is absent from PostgreSQL."""


class ProductRepository:
    """Read indexable ChedMed products from PostgreSQL with fixed SQL ownership."""

    _SELECT = """
        SELECT
            id::text AS id,
            COALESCE(NULLIF(title_fr, ''), NULLIF(title_en, ''), NULLIF(title_ar, '')) AS title,
            COALESCE(NULLIF(description_fr, ''), NULLIF(description_en, ''), NULLIF(description_ar, ''), '') AS description,
            COALESCE(category_id::text, 'non-categorise') AS category,
            price,
            discounted_price,
            COALESCE(NULLIF(status, ''), 'ACTIVE') AS status,
            is_sold,
            updated_at,
            thumbnail_url,
            title_fr,
            title_en,
            title_ar,
            description_fr,
            description_en,
            description_ar
        FROM public.products
        WHERE deleted_at IS NULL
          AND is_sold IS FALSE
          AND (status IS NULL OR lower(status) NOT IN ('inactive', 'inactif', 'disabled', 'deactivated', 'archived', 'draft'))
    """

    def __init__(self, database: PostgreSQLClient) -> None:
        """Store the injected PostgreSQL boundary."""
        self._database = database

    def get_all_products(self) -> list[Product]:
        """Return every active, non-sold, non-deleted product."""
        return self._read_products(self._SELECT + " ORDER BY updated_at ASC", ())

    def get_product(self, product_id: str) -> Product:
        """Return one indexable product by identifier.

        Raises:
            ValueError: If the identifier is blank.
            ProductNotFoundError: If the product is deleted, sold, inactive, or absent.
        """
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id ne peut pas être vide.")
        products = self._read_products(self._SELECT + " AND id::text = %s", (product_id.strip(),))
        if not products:
            raise ProductNotFoundError(f"Produit indexable introuvable : {product_id}.")
        return products[0]

    def get_products_updated_after(self, updated_after: datetime | str) -> list[Product]:
        """Return active products updated strictly after the supplied timestamp."""
        value = _updated_after_value(updated_after)
        return self._read_products(self._SELECT + " AND updated_at > %s ORDER BY updated_at ASC", (value,))

    def _read_products(self, query: str, params: tuple[Any, ...]) -> list[Product]:
        """Execute repository-owned SQL and map each returned row through Product."""
        try:
            with self._database.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
        except Exception as exc:
            LOGGER.exception("Lecture PostgreSQL de public.products impossible.")
            raise ProductRepositoryError("Impossible de lire les produits PostgreSQL.") from exc
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ProductRepositoryError("PostgreSQL a retourné des lignes produit invalides.")
        return [self._to_product(row) for row in rows]

    @staticmethod
    def _to_product(row: Mapping[str, Any]) -> Product:
        """Translate the known SQL row shape to the existing Product model."""
        payload = dict(row)
        payload["currency"] = "MAD"
        payload["imageUrls"] = [payload["thumbnail_url"]] if payload.get("thumbnail_url") else []
        multilingual_text = [
            payload.pop(column, None)
            for column in ("title_fr", "title_en", "title_ar", "description_fr", "description_en", "description_ar")
        ]
        payload["description"] = "\n".join(value for value in multilingual_text if isinstance(value, str) and value.strip()) or payload["description"]
        payload.pop("discounted_price", None)
        payload["updatedAt"] = _iso_datetime(payload.pop("updated_at"))
        payload["isSold"] = payload.pop("is_sold")
        payload.pop("thumbnail_url", None)
        try:
            return Product.from_dict(payload)
        except ProductValidationError as exc:
            raise ProductRepositoryError("Une ligne public.products ne respecte pas le modèle Product.") from exc


def _updated_after_value(updated_after: datetime | str) -> datetime | str:
    """Validate a repository timestamp filter without importing service concerns."""
    if isinstance(updated_after, datetime):
        if updated_after.tzinfo is None:
            raise ValueError("updated_after doit inclure un fuseau horaire.")
        return updated_after
    if isinstance(updated_after, str) and updated_after.strip():
        return updated_after.strip()
    raise ValueError("updated_after doit être une date ISO 8601 non vide.")


def _iso_datetime(value: Any) -> str:
    """Convert psycopg's timestamp value to Product's accepted ISO representation."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    raise ProductRepositoryError("updated_at PostgreSQL est invalide.")
