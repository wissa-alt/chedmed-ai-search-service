"""Unit tests for persistent product-oriented FAISS storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import Settings
from vector_store.faiss_manager import FAISSManager, FAISSManagerError


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Provide isolated index and mapping paths for every test."""
    return Settings(
        environment="test",
        host="127.0.0.1",
        port=5000,
        log_level="CRITICAL",
        db_host="127.0.0.1", db_port=5432, db_name="chedmed", db_user="test", db_password="password",
        chedmed_webhook_secret="test-webhook-secret",
        groq_api_key="test-groq-key",
        project_root=tmp_path,
    )


@pytest.fixture
def manager(settings: Settings) -> FAISSManager:
    """Return a manager whose temporary native and disk state is cleaned up."""
    instance = FAISSManager(settings)
    yield instance
    instance.clear()


def test_create_index_creates_empty_index(manager: FAISSManager) -> None:
    """An explicit index has no products until vectors are added."""
    manager.create_index(3)

    assert manager.count() == 0


def test_add_and_search_return_product_ids_not_vector_ids(manager: FAISSManager) -> None:
    """Search results expose stable product IDs and descending similarity scores."""
    first_vector_id = manager.add("product-a", np.array([1.0, 0.0, 0.0]))
    second_vector_id = manager.add("product-b", np.array([0.0, 1.0, 0.0]))

    results = manager.search(np.array([0.9, 0.1, 0.0]), top_k=2)

    assert first_vector_id == 0
    assert second_vector_id == 1
    assert [product_id for product_id, _ in results] == ["product-a", "product-b"]
    assert results[0][1] > results[1][1]
    assert manager.product_ids() == ("product-a", "product-b")


def test_save_and_load_restore_index_and_mappings(
    settings: Settings, manager: FAISSManager
) -> None:
    """A new manager restores the complete persisted vector state."""
    manager.add("product-a", np.array([1.0, 0.0]))
    manager.add("product-b", np.array([0.0, 1.0]))
    manager.save()

    restored = FAISSManager(settings)

    assert settings.faiss_index_path.is_file()
    assert settings.id_mapping_path.is_file()
    assert restored.count() == 2
    assert restored.search(np.array([1.0, 0.0]), top_k=1) == [("product-a", 1.0)]
    assert restored.add("product-c", np.array([1.0, 1.0])) == 2


def test_remove_deletes_vector_and_mapping(manager: FAISSManager) -> None:
    """Removing a product makes it absent from both store operations and search."""
    manager.add("product-a", np.array([1.0, 0.0]))
    manager.add("product-b", np.array([0.0, 1.0]))

    manager.remove("product-a")

    assert not manager.contains("product-a")
    assert manager.count() == 1
    assert manager.search(np.array([1.0, 0.0]), top_k=2) == [("product-b", 0.0)]


def test_update_replaces_embedding_and_keeps_vector_id(manager: FAISSManager) -> None:
    """Updating an item changes search ranking without duplicating its mapping."""
    product_a_vector_id = manager.add("product-a", np.array([1.0, 0.0]))
    manager.add("product-b", np.array([0.0, 1.0]))

    assert manager.update("product-a", np.array([0.1, 0.99])) == product_a_vector_id

    assert manager.count() == 2
    assert manager.search(np.array([0.0, 1.0]), top_k=2)[0][0] == "product-b"


def test_clear_removes_persisted_and_in_memory_state(
    settings: Settings, manager: FAISSManager
) -> None:
    """Clear erases the index, mapping files, and all active vectors."""
    manager.add("product-a", np.array([1.0, 0.0]))
    manager.save()

    manager.clear()

    assert manager.count() == 0
    assert not settings.faiss_index_path.exists()
    assert not settings.id_mapping_path.exists()
    assert manager.search(np.array([1.0, 0.0]), top_k=1) == []


def test_incompatible_dimensions_are_rejected(manager: FAISSManager) -> None:
    """All embeddings in one index must use the inferred first-vector dimension."""
    manager.add("product-a", np.array([1.0, 0.0]))

    with pytest.raises(FAISSManagerError, match="Dimension incompatible"):
        manager.add("product-b", np.array([1.0, 0.0, 0.0]))


def test_removing_absent_product_raises_explicit_error(manager: FAISSManager) -> None:
    """Callers cannot silently remove product IDs that are not indexed."""
    with pytest.raises(FAISSManagerError, match="n'est pas indexé"):
        manager.remove("unknown-product")


def test_duplicate_product_is_rejected(manager: FAISSManager) -> None:
    """One product may own only one vector mapping at a time."""
    manager.add("product-a", np.array([1.0, 0.0]))

    with pytest.raises(FAISSManagerError, match="déjà indexé"):
        manager.add("product-a", np.array([0.0, 1.0]))


def test_search_rejects_mapping_count_inconsistency(manager: FAISSManager) -> None:
    """Python validation prevents malformed state from reaching FAISS search."""
    manager.add("product-a", np.array([1.0, 0.0]))
    manager._product_to_vector["product-extra"] = 1

    with pytest.raises(FAISSManagerError, match="Nombre de vecteurs"):
        manager.search(np.array([1.0, 0.0]), top_k=1)


def test_search_rejects_index_id_inconsistency(manager: FAISSManager) -> None:
    """An index ID missing from mappings is rejected before native search."""
    manager.add("product-a", np.array([1.0, 0.0]))
    manager._vector_to_product = {4: "product-a"}
    manager._product_to_vector = {"product-a": 4}

    with pytest.raises(FAISSManagerError, match="Identifiants FAISS"):
        manager.search(np.array([1.0, 0.0]), top_k=1)
