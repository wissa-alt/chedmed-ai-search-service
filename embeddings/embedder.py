"""SentenceTransformer-based embedding service."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from config import Settings
from models.product import Product
from search.product_text_builder import ProductTextBuilder


LOGGER = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when embedding input, model loading, or encoding is invalid."""


class EmbeddingService:
    """
    Generate normalized embeddings through one isolated model runtime.

    Product-to-text conversion is delegated to ProductTextBuilder.

    This service is responsible only for:
    - validating embedding input;
    - applying E5 prefixes when necessary;
    - calling the embedding runtime;
    - validating and normalizing generated vectors.

    No product-specific text construction logic is kept here.
    """

    def __init__(
        self,
        settings: Settings,
        model: object | None = None,
        text_builder: ProductTextBuilder | None = None,
    ) -> None:
        self._model_name = settings.embedding_model_name
        self._device = _resolve_device(settings.embedding_device)
        self._uses_e5_prefixes = "e5" in self._model_name.lower()

        self._text_builder = text_builder or ProductTextBuilder()

        LOGGER.info(
            "Chargement du modèle d'embeddings %s sur %s.",
            self._model_name,
            self._device,
        )

        self._model = model
        self._runtime = None

        if model is None:
            try:
                from embeddings.isolated_runtime import (
                    IsolatedEmbeddingRuntime,
                )

                self._runtime = IsolatedEmbeddingRuntime(
                    self._model_name,
                    self._device,
                )

            except Exception as exc:
                LOGGER.exception(
                    "Impossible de démarrer le modèle "
                    "d'embeddings isolé."
                )

                raise EmbeddingServiceError(
                    "Le modèle d'embeddings est indisponible."
                ) from exc

    def close(self) -> None:
        """Release the isolated model worker."""

        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def embed(self, text: str) -> np.ndarray:
        """
        Generate one normalized embedding for arbitrary text.

        This method is useful for already prepared textual content.
        Product formatting must be performed outside this method.
        """

        if not isinstance(text, str) or not text.strip():
            raise EmbeddingServiceError(
                "Le texte à encoder ne peut pas être vide."
            )

        return self._encode_texts(
            [self._prefix_document(text.strip())]
        )[0]

    def embed_product(self, product: Product) -> np.ndarray:
        """Generate one normalized embedding for a product."""

        self._validate_product(product)

        document = self._text_builder.build(product)

        vectors = self._encode_documents([document])

        return vectors[0]

    def embed_products(
        self,
        products: Sequence[Product],
    ) -> list[np.ndarray]:
        """
        Generate normalized embeddings for multiple products in one batch.
        """

        product_list = list(products)

        if not product_list:
            return []

        for product in product_list:
            self._validate_product(product)

        documents = [
            self._text_builder.build(product)
            for product in product_list
        ]

        return self._encode_documents(documents)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate one normalized embedding for a search query."""

        if not isinstance(query, str) or not query.strip():
            raise EmbeddingServiceError(
                "La requête de recherche ne peut pas être vide."
            )

        return self._encode_texts(
            [self._prefix_query(query.strip())]
        )[0]

    def _encode_documents(
        self,
        documents: Sequence[str],
    ) -> list[np.ndarray]:
        """Encode product documents using the document/passages prefix."""

        prepared_documents = [
            self._prefix_document(document)
            for document in documents
        ]

        return self._encode_texts(prepared_documents)



    def _encode_texts(
        self,
        texts: Sequence[str],
    ) -> list[np.ndarray]:
        """Encode texts and return validated normalized vectors."""

        if not texts:
            return []

        try:
            if self._runtime is not None:
                LOGGER.debug(
                    "Encodage de %d texte(s) via le runtime isolé.",
                    len(texts),
                )

                encoded = self._runtime.encode(list(texts))

            else:
                assert self._model is not None

                encoded = self._model.encode(
                    list(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

        except Exception as exc:
            LOGGER.exception("Erreur model.encode().")

            raise EmbeddingServiceError(
                "SentenceTransformers n'a pas pu générer "
                "les embeddings."
            ) from exc

        vectors = np.asarray(
            encoded,
            dtype=np.float32,
        )

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(texts)
            or vectors.shape[1] == 0
        ):
            raise EmbeddingServiceError(
                "Le modèle a retourné des embeddings "
                "de forme invalide."
            )

        if not np.isfinite(vectors).all():
            raise EmbeddingServiceError(
                "Le modèle a retourné des embeddings non finis."
            )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise EmbeddingServiceError(
                "Le modèle a retourné un embedding nul."
            )

        normalized = vectors / norms

        return [
            vector.astype(
                np.float32,
                copy=False,
            )
            for vector in normalized
        ]

    def _prefix_document(
        self,
        document: str,
    ) -> str:
        """Apply the E5 passage prefix when required."""

        if self._uses_e5_prefixes:
            return f"passage: {document}"

        return document

    def _prefix_query(
        self,
        query: str,
    ) -> str:
        """Apply the E5 query prefix when required."""

        if self._uses_e5_prefixes:
            return f"query: {query}"

        return query

    @staticmethod
    def _validate_product(product: Product) -> None:
        """Validate that the value is a valid Product instance."""

        if not isinstance(product, Product):
            raise EmbeddingServiceError(
                "L'objet à encoder doit être un Product valide."
            )


def _resolve_device(
    configured_device: str,
) -> str:
    """
    Return a supported worker device without importing PyTorch locally.

    CPU remains the portable and deterministic default for
    catalogue synchronization.
    """

    device = configured_device.strip().lower()

    if (
        device in {"cpu", "cuda", "mps"}
        or device.startswith("cuda:")
    ):
        return device

    if device == "auto":
        return "cpu"

    raise EmbeddingServiceError(
        "EMBEDDING_DEVICE doit être cpu, mps, cuda, "
        "cuda:N ou auto."
    )