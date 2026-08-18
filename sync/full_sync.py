"""Command-line runner for one full ChedMed catalogue synchronisation."""

from __future__ import annotations

import logging

from config import get_settings
from database.catalogue_factory import create_catalogue_client
from embeddings.embedder import EmbeddingService
from services.sync_service import SynchronizationService
from vector_store.faiss_manager import FAISSManager
from search.product_text_builder import ProductTextBuilder

def main() -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    client = create_catalogue_client(settings)
    embedder: EmbeddingService | None = None
    try:
        embedder = EmbeddingService(settings)
        service = SynchronizationService(
            product_source=client,
            embedder=embedder,
            vector_store=FAISSManager(settings),
            text_builder=ProductTextBuilder(),
        )
        report = service.full_sync()
    finally:
        if embedder is not None:
            embedder.close()
        client.close()

    print(
        "Full sync terminé "
        f"(total={report.total_products}, ajoutés={report.indexed_products}, "
        f"mis_à_jour={report.updated_products}, ignorés={report.skipped_products}, "
        f"échecs={report.failed_products}, durée={report.duration_seconds:.2f}s)."
    )

    for failure in report.errors:
        print(f"- Échec produit {failure.product_id}: {failure.message}")

    return 0 if report.failed_products == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
