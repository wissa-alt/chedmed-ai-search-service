
"""Business orchestration for synchronising ChedMed products into FAISS."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from models.product import Product
from search.product_text_builder import ProductTextBuilder

LOGGER = logging.getLogger(__name__)


class ProductSourcePort(Protocol):

    def get_products_page(
        self,
        page: int,
        limit: int,
    ) -> list[Product]:
        """Return one page of current ChedMed products."""

    def get_product(
        self,
        product_id: str,
    ) -> Product:
        """Return one current product."""

    def get_products_updated_after(
        self,
        updated_after: datetime | str,
    ) -> list[Product]:
        """Return products modified after a timestamp."""

class ProductEmbeddingPort(Protocol):
    """Embedding operation required by synchronisation."""

    def embed_product(self, product: Product) -> np.ndarray:
        """Return an embedding for one product."""


class VectorStorePort(Protocol):
    """Vector-store operations required by synchronisation."""

    def contains(self, product_id: str) -> bool:
        """Return whether a product currently has a vector."""

    def add(self, product_id: str, embedding: np.ndarray) -> int:
        """Add a product vector and return its internal ID."""

    def update(self, product_id: str, embedding: np.ndarray) -> int:
        """Replace an existing product vector and return its internal ID."""

    def remove(self, product_id: str) -> None:
        """Remove one product vector."""

    def clear(self) -> None:
        """Remove the complete vector index and its mappings."""    

    def save(self) -> None:
        """Persist vectors and mappings."""


@dataclass(frozen=True, slots=True)
class SyncFailure:
    """A non-fatal product-level failure recorded during a batch sync."""

    product_id: str
    message: str


@dataclass(slots=True)
class SyncReport:
    """Summary and product-level failures from one completed sync operation."""

    total_products: int = 0
    indexed_products: int = 0
    updated_products: int = 0
    removed_products: int = 0
    skipped_products: int = 0
    failed_products: int = 0
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    finished_at: datetime | None = None
    duration_seconds: float = 0.0
    errors: list[SyncFailure] = field(default_factory=list)

    def finish(self) -> None:
        """Stamp completion time and calculate duration."""
        self.finished_at = datetime.now(timezone.utc)
        self.duration_seconds = (
            self.finished_at - self.started_at
        ).total_seconds()

    def record_failure(
        self,
        product_id: str,
        error: Exception,
    ) -> None:
        """Record one product failure without stopping the batch."""
        self.failed_products += 1
        self.errors.append(
            SyncFailure(
                product_id=product_id,
                message=str(error),
            )
        )


@dataclass(frozen=True, slots=True)
class ProductSyncResult:
    """Explicit outcome of synchronising or removing one product."""

    product_id: str
    action: str
    vector_id: int | None = None


class SynchronizationService:
    """Coordinate catalogue retrieval, text building, embeddings and FAISS."""

    def __init__(
        self,
        product_source: ProductSourcePort,
        embedder: ProductEmbeddingPort,
        vector_store: VectorStorePort,
        text_builder: ProductTextBuilder,
    ) -> None:
        """Store all dependencies through dependency injection.

        Args:
            product_source: Source of truth for ChedMed products.
            embedder: Product embedding generator.
            vector_store: FAISS-backed vector store.
            text_builder: Builder used to create deterministic product text.
        """
        self._product_source = product_source
        self._embedder = embedder
        self._vector_store = vector_store
        self._text_builder = text_builder

    def full_sync(self,
    page_size: int = 500,) -> SyncReport:
        """Synchronise the complete ChedMed catalogue page by page.

    The existing vector store is cleared first so that the resulting
    FAISS index exactly represents the current catalogue state.

    Args:
        page_size: Maximum number of products requested per page.

    Returns:
        A report describing the completed synchronization.

    Raises:
        ValueError: If page_size is not strictly positive.
    """
        if not isinstance(page_size, int) or page_size < 1:
            raise ValueError("page_size doit être un entier strictement positif.")

        report = SyncReport()

        LOGGER.info(
            "Début de la synchronisation complète du catalogue : page_size=%d.",
            page_size,
        )
         # A full synchronization must rebuild the vector store completely.
        self._vector_store.clear()

        page = 1
        while True:
            LOGGER.info(
                "Récupération de la page catalogue %d.",
                page,
            )

            products = self._product_source.get_products_page(
                page,
                page_size,
            )

            if not products:
                LOGGER.info(
                    "Fin du catalogue atteinte à la page %d.",
                    page,
                )
                break

            LOGGER.info(
                "Page %d récupérée : %d produit(s).",
                page,
                len(products),
            )

            report.total_products += len(products)
            for product in products:
                self._sync_product_for_report(
                    product,
                    report,
                )

            # If the source returns fewer products than requested,
            # this is the final page.
            if len(products) < page_size:
                LOGGER.info(
                    "Dernière page détectée : page=%d, produits=%d.",
                    page,
                    len(products),
                )
                break

            page += 1

        self._vector_store.save()

        report.finish()

        LOGGER.info(
            "Synchronisation complète terminée : "
            "%d ajouté(s), %d échec(s), durée=%.2fs.",
            report.indexed_products,
            report.failed_products,
            report.duration_seconds,
        )

        return report

    def sync_product(
        self,
        product_id: str,
    ) -> ProductSyncResult:
        """Fetch and synchronise one current product."""
        product = self._product_source.get_product(product_id)

        result = self._sync_product_model(product)

        self._vector_store.save()

        return result

    def remove_product(
        self,
        product_id: str,
    ) -> ProductSyncResult:
        """Remove one indexed product and persist the change."""
        if not self._vector_store.contains(product_id):
            LOGGER.info(
                "Produit %s absent de l'index : suppression ignorée.",
                product_id,
            )

            return ProductSyncResult(
                product_id=product_id,
                action="skipped",
            )

        self._vector_store.remove(product_id)
        self._vector_store.save()

        LOGGER.info(
            "Produit %s supprimé du catalogue vectoriel.",
            product_id,
        )

        return ProductSyncResult(
            product_id=product_id,
            action="removed",
        )

    def reactivate_product(
        self,
        product_id: str,
    ) -> ProductSyncResult:
        """Synchronise a reactivated product from the source of truth."""
        return self.sync_product(product_id)

    def sync_updated_products(
        self,
        updated_after: datetime | str,
    ) -> SyncReport:
        """Synchronise products modified after a timestamp."""
        report = SyncReport()

        LOGGER.info(
            "Début de la synchronisation incrémentale."
        )

        products = self._product_source.get_products_updated_after(
            updated_after
        )

        report.total_products = len(products)

        for product in products:
            self._sync_product_for_report(
                product,
                report,
            )

        self._vector_store.save()

        report.finish()

        LOGGER.info(
            "Synchronisation incrémentale terminée : "
            "%d ajouté(s), %d mis à jour, %d échec(s).",
            report.indexed_products,
            report.updated_products,
            report.failed_products,
        )

        return report

    def _sync_product_for_report(
        self,
        product: Product,
        report: SyncReport,
    ) -> None:
        """Synchronise one product and record failures without stopping."""
        try:
            result = self._sync_product_model(product)

        except Exception as exc:
            product_id = (
                product.id
                if isinstance(product, Product)
                else "unknown-product"
            )

            LOGGER.exception(
                "Échec de synchronisation du produit %s.",
                product_id,
            )

            report.record_failure(
                product_id,
                exc,
            )

            return

        if result.action == "added":
            report.indexed_products += 1

        elif result.action == "updated":
            report.updated_products += 1

        else:
            report.skipped_products += 1

    def _sync_product_model(
        self,
        product: Product,
    ) -> ProductSyncResult:
        """Build the semantic document, embed it and update FAISS."""
        if not isinstance(product, Product):
            raise TypeError(
                "La synchronisation requiert un objet Product valide."
            )

        # Build the deterministic semantic representation.
        document = self._text_builder.build(product)

        if not document.strip():
            raise ValueError(
                f"Impossible de générer un document sémantique "
                f"pour le produit {product.id}."
            )

        # The current embedding interface works with Product objects.
        # The text builder is therefore kept as the canonical semantic
        # representation layer without changing the existing embedder API.
        embedding = self._embedder.embed_product(product)

        exists = self._vector_store.contains(product.id)

        if exists:
            vector_id = self._vector_store.update(
                product.id,
                embedding,
            )

            LOGGER.info(
                "Produit %s mis à jour dans l'index vectoriel.",
                product.id,
            )

            return ProductSyncResult(
                product_id=product.id,
                action="updated",
                vector_id=vector_id,
            )

        vector_id = self._vector_store.add(
            product.id,
            embedding,
        )

        LOGGER.info(
            "Produit %s ajouté dans l'index vectoriel.",
            product.id,
        )

        return ProductSyncResult(
            product_id=product.id,
            action="added",
            vector_id=vector_id,
        )
