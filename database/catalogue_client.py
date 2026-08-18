"""Source-agnostic contract for ChedMed product catalogues."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from models.product import Product


class CatalogueClient(Protocol):
    """Provide typed products regardless of the backing transport."""

    def get_all_products(self) -> list[Product]:
        """Return every indexable product."""

    def get_products_page(self, page: int, limit: int) -> list[Product]:
        """Return one one-based product page."""

    def get_product(self, product_id: str) -> Product:
        """Return one product by identifier."""

    def get_products_updated_after(self, updated_after: datetime | str) -> list[Product]:
        """Return products updated after a timestamp."""

    def close(self) -> None:
        """Release source-specific resources."""
