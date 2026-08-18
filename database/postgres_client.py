"""PostgreSQL implementation of the historical ChedMed catalogue client port."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row

from config import Settings
from models.product import Product, ProductValidationError

LOGGER = logging.getLogger(__name__)


class PostgresCatalogueError(RuntimeError):
    """Base exception for PostgreSQL catalogue access failures."""


class PostgresConnectionError(PostgresCatalogueError):
    """Raised when the reusable PostgreSQL connection cannot be opened."""


class PostgresQueryError(PostgresCatalogueError):
    """Raised when a catalogue SQL query cannot be completed."""


class PostgresProductNotFoundError(PostgresCatalogueError):
    """Raised when the requested catalogue product is absent."""


class ConnectionPort(Protocol):
    """Minimal connection interface used by the catalogue client."""

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        """Return a context-managed database cursor."""

    def close(self) -> None:
        """Close the database connection."""


ConnectionFactory = Callable[..., ConnectionPort]


class PostgresCatalogueClient:
    """Read ChedMed products from PostgreSQL through the legacy client API."""

    _SELECT = """
        SELECT
            p.id::text AS id,
            COALESCE(NULLIF(p.title_fr, ''), NULLIF(p.title_en, ''), NULLIF(p.title_ar, '')) AS title,
            concat_ws(E'\n', NULLIF(p.title_fr, ''), NULLIF(p.title_en, ''), NULLIF(p.title_ar, ''),
                NULLIF(p.description_fr, ''), NULLIF(p.description_en, ''), NULLIF(p.description_ar, ''),
                'Prix: ' || p.price::text || ' MAD',
                CASE WHEN p.discounted_price IS NULL THEN NULL ELSE 'Prix remisé: ' || p.discounted_price::text || ' MAD' END) AS description,
            COALESCE(NULLIF(category.title_fr, ''), NULLIF(category.title_en, ''), NULLIF(category.title_ar, ''), p.category_id::text) AS category,
            COALESCE(NULLIF(brand.title_fr, ''), NULLIF(brand.title_en, ''), NULLIF(brand.title_ar, '')) AS brand,
            COALESCE(NULLIF(color.color_fr, ''), NULLIF(color.color_en, ''), NULLIF(color.color_ar, '')) AS color,
            COALESCE(NULLIF(condition.title_fr, ''), NULLIF(condition.title_en, ''), NULLIF(condition.title_ar, '')) AS condition,
            p.price, 'MAD'::text AS currency,
            COALESCE(NULLIF(city.city_title_fr, ''), NULLIF(city.city_title_en, ''), NULLIF(city.city_title_ar, '')) AS city,
            p.thumbnail_url,
            COALESCE((SELECT array_agg(image.image_url ORDER BY image.id)
                      FROM public.prod_images AS image
                      WHERE image.product_id = p.id AND image.deleted_at IS NULL), ARRAY[]::varchar[]) AS image_urls,
            p.status::text AS status, p.is_sold AS "isSold", p.updated_at AS "updatedAt"
        FROM public.products AS p
        LEFT JOIN public.categories AS category ON category.id = p.category_id AND category.deleted_at IS NULL
        LEFT JOIN public.brands AS brand ON brand.id = p.brand_id
        LEFT JOIN public.colors AS color ON color.id::text = p.color_id
        LEFT JOIN public.product_conditions AS condition ON condition.id = p.condition_id
        LEFT JOIN public.addresses AS address ON address.id = p.address_id
        LEFT JOIN public.referential_city AS city ON city.id = address.city_id
        WHERE p.deleted_at IS NULL AND p.is_sold IS FALSE AND p.status = 'ACCEPTED'::public.enum_products_status
    """
    _SELECT_ALL = _SELECT.replace(
        "WHERE p.deleted_at IS NULL AND p.is_sold IS FALSE AND p.status = 'ACCEPTED'::public.enum_products_status",
        "WHERE p.deleted_at IS NULL",
    )

    def __init__(self, settings: Settings, connection_factory: ConnectionFactory | None = None) -> None:
        """Open one reusable injected or psycopg-managed PostgreSQL connection."""
        factory = connection_factory or psycopg.connect
        try:
            self._connection = factory(settings.postgres_dsn, row_factory=dict_row)
        except Exception as exc:
            LOGGER.exception("Connexion PostgreSQL au catalogue impossible.")
            raise PostgresConnectionError("Impossible de se connecter au catalogue PostgreSQL.") from exc

    def close(self) -> None:
        """Close the reusable PostgreSQL connection."""
        self._connection.close()

    def __enter__(self) -> "PostgresCatalogueClient":
        """Return the open catalogue client."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection after the caller leaves the context."""
        self.close()

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        """Expose a cursor for repository compatibility with translated errors."""
        try:
            with self._connection.cursor() as cursor:
                yield cursor
        except PostgresCatalogueError:
            raise
        except Exception as exc:
            LOGGER.exception("Opération PostgreSQL impossible.")
            raise PostgresQueryError("L'opération PostgreSQL a échoué.") from exc

    def get_products_page(self, page: int, limit: int) -> list[Product]:
        """Return one one-based page of indexable products."""
        if not isinstance(page, int) or page < 1 or not isinstance(limit, int) or limit < 1:
            raise ValueError("page et limit doivent être strictement positifs.")
        return self._read_products(self._SELECT + " ORDER BY p.updated_at ASC, p.id ASC LIMIT %s OFFSET %s", (limit, (page - 1) * limit))

    def get_products_page_any_status(self, page: int, limit: int) -> list[Product]:
        """Return a page of all non-deleted products for an explicit full index."""
        if not isinstance(page, int) or page < 1 or not isinstance(limit, int) or limit < 1:
            raise ValueError("page et limit doivent être strictement positifs.")
        return self._read_products(
            self._SELECT_ALL + " ORDER BY p.updated_at ASC, p.id ASC LIMIT %s OFFSET %s",
            (limit, (page - 1) * limit),
        )

    def get_all_products(self) -> list[Product]:
        """Return every indexable product through the legacy client contract."""
        return self._read_products(self._SELECT + " ORDER BY p.updated_at ASC, p.id ASC", ())

    def get_product(self, product_id: str) -> Product:
        """Return a single active, non-sold product by identifier."""
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id ne peut pas être vide.")
        products = self._read_products(self._SELECT + " AND p.id::text = %s", (product_id.strip(),))
        if not products:
            raise PostgresProductNotFoundError(f"Produit introuvable : {product_id}.")
        return products[0]

    def get_product_any_status(self, product_id: str) -> Product:
        """Return one non-deleted product for catalogue analysis/full search."""
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id ne peut pas être vide.")
        products = self._read_products(
            self._SELECT_ALL + " AND p.id::text = %s", (product_id.strip(),)
        )
        if not products:
            raise PostgresProductNotFoundError(f"Produit introuvable : {product_id}.")
        return products[0]

    def get_all_products_any_status(self) -> list[Product]:
        """Return all non-deleted products for diagnostics or a full reindex."""
        return self._read_products(
            self._SELECT_ALL + " ORDER BY p.updated_at ASC, p.id ASC", ()
        )

    def count_all_products(self) -> int:
        """Count all non-deleted PostgreSQL products for sync diagnostics."""
        try:
            with self.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM public.products WHERE deleted_at IS NULL"
                )
                row = cursor.fetchone()
        except Exception as exc:
            raise PostgresQueryError("Impossible de compter le catalogue PostgreSQL.") from exc
        if isinstance(row, Mapping):
            return int(next(iter(row.values())))
        return int(row[0])

    def get_products_updated_after(self, updated_after: datetime | str) -> list[Product]:
        """Return indexable products modified strictly after a timestamp."""
        return self._read_products(self._SELECT + " AND p.updated_at > %s ORDER BY p.updated_at ASC, p.id ASC", (_timestamp_value(updated_after),))

    def get_products_updated_after_any_status(
        self, updated_after: datetime | str
    ) -> list[Product]:
        """Return all non-deleted products modified after a timestamp."""
        return self._read_products(
            self._SELECT_ALL
            + " AND p.updated_at > %s ORDER BY p.updated_at ASC, p.id ASC",
            (_timestamp_value(updated_after),),
        )

    def _read_products(self, query: str, params: tuple[Any, ...]) -> list[Product]:
        """Execute one query and map its rows with Product.from_dict."""
        try:
            with self.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
        except PostgresCatalogueError:
            raise
        except Exception as exc:
            LOGGER.exception("Lecture des produits PostgreSQL impossible.")
            raise PostgresQueryError("Impossible de lire les produits PostgreSQL.") from exc
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise PostgresQueryError("PostgreSQL a retourné des lignes produit invalides.")
        return [self._to_product(row) for row in rows]

    @staticmethod
    def _to_product(row: Mapping[str, Any]) -> Product:
        """Map one explicit SQL row to the existing immutable Product model."""
        payload = dict(row)
        thumbnail_url = payload.pop("thumbnail_url", None)
        image_urls = payload.pop("image_urls", [])
        if not isinstance(image_urls, list):
            raise PostgresQueryError("Les images PostgreSQL doivent être une liste.")
        if isinstance(thumbnail_url, str) and thumbnail_url.strip() and thumbnail_url not in image_urls:
            image_urls.insert(0, thumbnail_url)
        payload["imageUrls"] = image_urls
        if isinstance(payload.get("updatedAt"), datetime):
            payload["updatedAt"] = payload["updatedAt"].isoformat()
        try:
            return Product.from_dict(payload)
        except ProductValidationError as exc:
            raise PostgresQueryError("Une ligne PostgreSQL ne respecte pas le modèle Product.") from exc


PostgreSQLClient = PostgresCatalogueClient
PostgreSQLClientError = PostgresCatalogueError


def _timestamp_value(updated_after: datetime | str) -> datetime | str:
    """Validate a timestamp accepted by the historical catalogue interface."""
    if isinstance(updated_after, datetime):
        if updated_after.tzinfo is None:
            raise ValueError("updated_after doit inclure un fuseau horaire.")
        return updated_after
    if isinstance(updated_after, str) and updated_after.strip():
        return updated_after.strip()
    raise ValueError("updated_after doit être une date ISO 8601 non vide.")
