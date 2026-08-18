"""Semantic embedding generation independent of search storage and HTTP."""

from embeddings.embedder import EmbeddingService, EmbeddingServiceError

__all__ = ["EmbeddingService", "EmbeddingServiceError"]
