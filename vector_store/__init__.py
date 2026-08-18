"""Persistent FAISS vector storage isolated from application services."""

from vector_store.faiss_manager import FAISSManager, FAISSManagerError

__all__ = ["FAISSManager", "FAISSManagerError"]
