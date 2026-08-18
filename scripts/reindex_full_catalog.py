"""Explicitly rebuild FAISS with every non-deleted PostgreSQL product status."""

from __future__ import annotations

from config import get_settings
from database.catalogue_factory import create_catalogue_client
from embeddings.embedder import EmbeddingService
from search.product_text_builder import ProductTextBuilder
from services.sync_service import SynchronizationService
from vector_store.faiss_manager import FAISSManager


def main() -> int:
    settings = get_settings()
    catalogue = create_catalogue_client(settings)
    embedder = EmbeddingService(settings, text_builder=ProductTextBuilder())
    try:
        vectors = FAISSManager(settings)
        report = SynchronizationService(
            catalogue, embedder, vectors, ProductTextBuilder()
        ).full_sync(include_all=True)
        print({
            "catalogue_products": report.total_products,
            "indexed_products": report.indexed_products,
            "failed_products": report.failed_products,
            "faiss_ntotal": vectors.count(),
        })
        return 1 if report.failed_products else 0
    finally:
        embedder.close()
        catalogue.close()


if __name__ == "__main__":
    raise SystemExit(main())
