"""Compare non-deleted PostgreSQL IDs with the durable FAISS mapping."""

from __future__ import annotations

from config import get_settings
from database.catalogue_factory import create_catalogue_client
from vector_store.faiss_manager import FAISSManager


def main() -> int:
    settings = get_settings()
    catalogue = create_catalogue_client(settings)
    try:
        postgres_ids = {
            product.id for product in catalogue.get_all_products_any_status()
        }
        vectors = FAISSManager(settings)
        mapping_ids = set(vectors.product_ids())
        missing_ids = sorted(postgres_ids - mapping_ids)
        extra_ids = sorted(mapping_ids - postgres_ids)
        report = {
            "postgres_non_deleted_count": len(postgres_ids),
            "mapping_count": len(mapping_ids),
            "faiss_ntotal": vectors.count(),
            "missing_ids": missing_ids,
            "extra_ids": extra_ids,
        }
        print(report)
        return 0 if not missing_ids and not extra_ids and len(postgres_ids) == vectors.count() else 1
    finally:
        catalogue.close()


if __name__ == "__main__":
    raise SystemExit(main())
