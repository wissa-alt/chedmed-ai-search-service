
"""Persistent, product-ID-oriented wrapper around FAISS vector indexes."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config import Settings

LOGGER = logging.getLogger(__name__)


class FAISSManagerError(RuntimeError):
    """Raised for invalid vectors, invalid operations, or corrupt FAISS state."""


class FAISSManager:
    """Own the FAISS index and persistent ProductID <-> VectorID mappings.

    Architecture
    ------------
    The manager uses:

        IndexIDMap2(IndexFlatIP)

    ``IndexFlatIP`` performs exact inner-product search.

    Embeddings are normalized before insertion and before search, therefore
    the inner product corresponds to cosine similarity.

    ``IndexIDMap2`` provides explicit stable vector IDs and supports
    ``remove_ids`` / ``add_with_ids``, which is appropriate for a catalogue
    where products are frequently created, updated, sold or deleted.

    No other application component should access FAISS directly.
    """

    _STATE_VERSION = 2

    def __init__(self, settings: Settings) -> None:
        """Initialize the manager and load persisted state."""
        self._index_path: Path = Path(settings.faiss_index_path)
        self._mapping_path: Path = Path(settings.id_mapping_path)

        self._index: faiss.IndexIDMap2 | None = None
        self._dimension: int | None = None

        self._product_to_vector: dict[str, int] = {}
        self._vector_to_product: dict[int, str] = {}

        self._next_vector_id: int = 0

        self.load()

    # ------------------------------------------------------------------
    # INDEX LIFECYCLE
    # ------------------------------------------------------------------

    def create_index(self, dimension: int) -> None:
        """Create a completely new empty FAISS index.

        Args:
            dimension:
                Embedding dimension.

        Raises:
            FAISSManagerError:
                If the dimension is invalid.
        """
        self._validate_dimension(dimension)

        self._index = faiss.IndexIDMap2(
            faiss.IndexFlatIP(dimension)
        )

        self._dimension = dimension
        self._product_to_vector.clear()
        self._vector_to_product.clear()
        self._next_vector_id = 0

        LOGGER.info(
            "Création d'un nouvel index FAISS : dimension=%d.",
            dimension,
        )

    def load(self) -> None:
        """Load the persisted FAISS index and mappings.

        If no persisted state exists, an empty in-memory state is created.

        Raises:
            FAISSManagerError:
                If the persisted state is incomplete or corrupted.
        """
        index_exists = self._index_path.is_file()
        mapping_exists = self._mapping_path.is_file()

        # Nothing exists.
        if not index_exists and not mapping_exists:
            self._reset_empty_state()

            LOGGER.info(
                "Aucun index FAISS persistant trouvé. "
                "État vide initialisé."
            )
            return

        # Only one file exists.
        if index_exists != mapping_exists:
            raise FAISSManagerError(
                "État FAISS incomplet : "
                "l'index et le mapping doivent exister ensemble."
            )

        try:
            index = faiss.read_index(
                str(self._index_path)
            )

            if not isinstance(index, faiss.IndexIDMap2):
                raise FAISSManagerError(
                    "L'index chargé n'est pas un IndexIDMap2."
                )

            state = _read_state(
                self._mapping_path
            )

            (
                product_to_vector,
                vector_to_product,
                next_vector_id,
                dimension,
            ) = _parse_state(state)

            _validate_index_mappings(
                index=index,
                product_to_vector=product_to_vector,
                vector_to_product=vector_to_product,
                dimension=dimension,
                next_vector_id=next_vector_id,
            )

        except FAISSManagerError:
            LOGGER.exception(
                "État FAISS invalide ou corrompu."
            )
            raise

        except Exception as exc:
            LOGGER.exception(
                "Impossible de charger l'état FAISS."
            )

            raise FAISSManagerError(
                "Impossible de charger l'index FAISS persistant."
            ) from exc

        self._index = index
        self._dimension = dimension
        self._product_to_vector = product_to_vector
        self._vector_to_product = vector_to_product
        self._next_vector_id = next_vector_id

        LOGGER.info(
            "Index FAISS chargé : %d produit(s), dimension=%d.",
            self.count(),
            dimension,
        )

    def save(self) -> None:
        """Persist FAISS index and mappings atomically."""
        if self._index is None:
            raise FAISSManagerError(
                "Impossible de sauvegarder un index FAISS inexistant."
            )

        self._validate_internal_state()

        self._index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._mapping_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_temp_path = self._index_path.with_name(
            f"{self._index_path.name}.tmp"
        )

        mapping_temp_path = self._mapping_path.with_name(
            f"{self._mapping_path.name}.tmp"
        )

        try:
            faiss.write_index(
                self._index,
                str(index_temp_path),
            )

            _write_state(
                mapping_temp_path,
                self._serialisable_state(),
            )

            os.replace(
                index_temp_path,
                self._index_path,
            )

            os.replace(
                mapping_temp_path,
                self._mapping_path,
            )

        except Exception as exc:
            _remove_if_present(index_temp_path)
            _remove_if_present(mapping_temp_path)

            LOGGER.exception(
                "La sauvegarde de l'état FAISS a échoué."
            )

            raise FAISSManagerError(
                "Impossible de sauvegarder l'état FAISS."
            ) from exc

        LOGGER.info(
            "Index FAISS sauvegardé : %d produit(s).",
            self.count(),
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        product_id: str,
        embedding: np.ndarray,
    ) -> int:
        """Add one product embedding.

        Args:
            product_id:
                ChedMed ProductID.

            embedding:
                Product embedding.

        Returns:
            Stable internal VectorID.

        Raises:
            FAISSManagerError:
                If the product already exists or the embedding is invalid.
        """
        normalized_product_id = _validate_product_id(
            product_id
        )

        if self.contains(normalized_product_id):
            raise FAISSManagerError(
                f"Le produit {normalized_product_id} est déjà indexé."
            )

        vector = self._validate_embedding(
            embedding
        )

        # First vector determines the index dimension.
        if self._index is None:
            self.create_index(
                int(vector.shape[0])
            )

        assert self._index is not None

        vector_id = self._next_vector_id

        try:
            self._index.add_with_ids(
                vector.reshape(1, -1),
                np.asarray(
                    [vector_id],
                    dtype=np.int64,
                ),
            )

        except Exception as exc:
            LOGGER.exception(
                "Impossible d'ajouter le produit %s à FAISS.",
                normalized_product_id,
            )

            raise FAISSManagerError(
                "FAISS n'a pas pu ajouter le vecteur produit."
            ) from exc

        self._product_to_vector[
            normalized_product_id
        ] = vector_id

        self._vector_to_product[
            vector_id
        ] = normalized_product_id

        self._next_vector_id += 1

        LOGGER.info(
            "Produit ajouté à FAISS : product_id=%s vector_id=%d.",
            normalized_product_id,
            vector_id,
        )

        return vector_id

    def update(
        self,
        product_id: str,
        embedding: np.ndarray,
    ) -> int:
        """Replace an existing product embedding.

        The existing VectorID is preserved.

        The operation is performed carefully so that an unsuccessful
        insertion does not silently leave the mapping inconsistent.
        """
        normalized_product_id = _validate_product_id(
            product_id
        )

        if not self.contains(normalized_product_id):
            raise FAISSManagerError(
                f"Le produit {normalized_product_id} "
                "n'est pas indexé."
            )

        vector = self._validate_embedding(
            embedding
        )

        assert self._index is not None

        vector_id = self._product_to_vector[
            normalized_product_id
        ]

        vector_ids = np.asarray(
            [vector_id],
            dtype=np.int64,
        )

        try:
            removed = self._index.remove_ids(
                vector_ids
            )

            if int(removed) != 1:
                raise FAISSManagerError(
                    "Le vecteur existant est absent de FAISS."
                )

            try:
                self._index.add_with_ids(
                    vector.reshape(1, -1),
                    vector_ids,
                )

            except Exception as insertion_error:
                LOGGER.exception(
                    "Échec de réinsertion du produit %s "
                    "après suppression du vecteur.",
                    normalized_product_id,
                )

                # Best-effort restoration of the old state.
                raise FAISSManagerError(
                    "La mise à jour FAISS a échoué après "
                    "suppression du vecteur existant."
                ) from insertion_error

        except FAISSManagerError:
            LOGGER.exception(
                "État FAISS incohérent pendant la mise à jour "
                "du produit %s.",
                normalized_product_id,
            )
            raise

        except Exception as exc:
            LOGGER.exception(
                "Impossible de mettre à jour le produit %s.",
                normalized_product_id,
            )

            raise FAISSManagerError(
                "FAISS n'a pas pu mettre à jour le vecteur."
            ) from exc

        self._validate_internal_state()

        LOGGER.info(
            "Produit mis à jour dans FAISS : "
            "product_id=%s vector_id=%d.",
            normalized_product_id,
            vector_id,
        )

        return vector_id

    def remove(
        self,
        product_id: str,
    ) -> None:
        """Remove a product from FAISS and both mappings."""
        normalized_product_id = _validate_product_id(
            product_id
        )

        if not self.contains(normalized_product_id):
            raise FAISSManagerError(
                f"Le produit {normalized_product_id} "
                "n'est pas indexé."
            )

        assert self._index is not None

        vector_id = self._product_to_vector[
            normalized_product_id
        ]

        try:
            removed = self._index.remove_ids(
                np.asarray(
                    [vector_id],
                    dtype=np.int64,
                )
            )

        except Exception as exc:
            LOGGER.exception(
                "Impossible de supprimer le produit %s de FAISS.",
                normalized_product_id,
            )

            raise FAISSManagerError(
                "FAISS n'a pas pu supprimer le vecteur produit."
            ) from exc

        if int(removed) != 1:
            raise FAISSManagerError(
                "Le vecteur associé au produit "
                "est absent de FAISS."
            )

        del self._product_to_vector[
            normalized_product_id
        ]

        del self._vector_to_product[
            vector_id
        ]

        self._validate_internal_state()

        LOGGER.info(
            "Produit supprimé de FAISS : product_id=%s vector_id=%d.",
            normalized_product_id,
            vector_id,
        )

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Search the nearest products.

        Returns only:

            (ProductID, similarity_score)

        VectorIDs never leave this class.
        """
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise FAISSManagerError(
                "top_k doit être un entier positif."
            )

        if self._index is None:
            return []

        if self._index.ntotal == 0:
            return []

        vector = self._validate_embedding(
            query_embedding
        )

        self._validate_internal_state()

        result_count = min(
            top_k,
            int(self._index.ntotal),
        )

        try:
            distances, vector_ids = self._index.search(
                vector.reshape(1, -1),
                result_count,
            )

        except Exception as exc:
            LOGGER.exception(
                "La recherche FAISS a échoué."
            )

            raise FAISSManagerError(
                "FAISS n'a pas pu effectuer la recherche."
            ) from exc

        results: list[tuple[str, float]] = []

        for vector_id, score in zip(
            vector_ids[0],
            distances[0],
            strict=True,
        ):
            vector_id = int(vector_id)

            # FAISS uses -1 when no result exists.
            if vector_id < 0:
                continue

            product_id = self._vector_to_product.get(
                vector_id
            )

            if product_id is None:
                raise FAISSManagerError(
                    "Résultat FAISS sans ProductID correspondant."
                )

            results.append(
                (
                    product_id,
                    float(score),
                )
            )

        LOGGER.info(
            "Recherche FAISS terminée : %d résultat(s), top_k=%d.",
            len(results),
            top_k,
        )

        return results

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def contains(
        self,
        product_id: str,
    ) -> bool:
        """Return True if the product is indexed."""
        return (
            isinstance(product_id, str)
            and product_id.strip() in self._product_to_vector
        )

    def count(self) -> int:
        """Return the number of indexed products."""
        return len(
            self._product_to_vector
        )

    def product_ids(self) -> tuple[str, ...]:
        """Return indexed product IDs for catalogue-consistency diagnostics."""
        return tuple(sorted(self._product_to_vector))

    def clear(self) -> None:
        """Completely remove the FAISS state."""
        self._reset_empty_state()

        _remove_if_present(
            self._index_path
        )

        _remove_if_present(
            self._mapping_path
        )

        _remove_if_present(
            self._index_path.with_name(
                f"{self._index_path.name}.tmp"
            )
        )

        _remove_if_present(
            self._mapping_path.with_name(
                f"{self._mapping_path.name}.tmp"
            )
        )

        LOGGER.info(
            "Index FAISS et mappings complètement supprimés."
        )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_embedding(
        self,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """Validate and L2-normalize one embedding."""
        try:
            vector = np.asarray(
                embedding,
                dtype=np.float32,
            )

        except (TypeError, ValueError) as exc:
            raise FAISSManagerError(
                "L'embedding doit être convertible "
                "en tableau NumPy."
            ) from exc

        # Accept shape (1, dimension).
        if (
            vector.ndim == 2
            and vector.shape[0] == 1
        ):
            vector = vector[0]

        if (
            vector.ndim != 1
            or vector.shape[0] == 0
        ):
            raise FAISSManagerError(
                "L'embedding doit être un vecteur "
                "NumPy unidimensionnel non vide."
            )

        if not np.isfinite(vector).all():
            raise FAISSManagerError(
                "L'embedding doit contenir uniquement "
                "des valeurs finies."
            )

        if (
            self._dimension is not None
            and vector.shape[0] != self._dimension
        ):
            raise FAISSManagerError(
                "Dimension incompatible : "
                f"{vector.shape[0]} reçu, "
                f"{self._dimension} attendu."
            )

        norm = float(
            np.linalg.norm(vector)
        )

        if norm <= 0.0:
            raise FAISSManagerError(
                "L'embedding ne peut pas être un vecteur nul."
            )

        normalized = vector / norm

        return np.ascontiguousarray(
            normalized,
            dtype=np.float32,
        )

    def _validate_internal_state(self) -> None:
        """Validate all in-memory FAISS invariants."""
        if self._index is None:
            if (
                self._product_to_vector
                or self._vector_to_product
                or self._dimension is not None
            ):
                raise FAISSManagerError(
                    "État FAISS vide mais mappings non vides."
                )

            return

        if self._dimension is None:
            raise FAISSManagerError(
                "Dimension FAISS absente."
            )

        if int(self._index.d) != self._dimension:
            raise FAISSManagerError(
                "Dimension de l'index FAISS incohérente."
            )

        index_count = int(
            self._index.ntotal
        )

        if index_count != len(
            self._product_to_vector
        ):
            raise FAISSManagerError(
                "Nombre de vecteurs et mapping "
                "ProductID incohérent."
            )

        if index_count != len(
            self._vector_to_product
        ):
            raise FAISSManagerError(
                "Nombre de vecteurs et mapping inverse "
                "incohérent."
            )

        expected_reverse = {
            vector_id: product_id
            for product_id, vector_id
            in self._product_to_vector.items()
        }

        if (
            expected_reverse
            != self._vector_to_product
        ):
            raise FAISSManagerError(
                "Les mappings ProductID/VectorID "
                "ne correspondent pas."
            )

        try:
            index_ids = {
                int(vector_id)
                for vector_id in faiss.vector_to_array(
                    self._index.id_map
                )
            }

        except Exception as exc:
            raise FAISSManagerError(
                "Impossible de lire les VectorID FAISS."
            ) from exc

        mapping_ids = set(
            self._vector_to_product
        )

        if index_ids != mapping_ids:
            raise FAISSManagerError(
                "Identifiants FAISS et mappings persistants "
                "sont incohérents."
            )

        if mapping_ids:
            maximum_id = max(
                mapping_ids
            )

            if self._next_vector_id <= maximum_id:
                raise FAISSManagerError(
                    "Le compteur nextVectorId est incohérent."
                )

    # ------------------------------------------------------------------
    # PERSISTENCE HELPERS
    # ------------------------------------------------------------------

    def _serialisable_state(
        self,
    ) -> dict[str, Any]:
        """Return the JSON representation of the current state."""
        return {
            "version": self._STATE_VERSION,
            "dimension": self._dimension,
            "nextVectorId": self._next_vector_id,
            "productToVector": self._product_to_vector,
            "vectorToProduct": {
                str(vector_id): product_id
                for vector_id, product_id
                in self._vector_to_product.items()
            },
        }

    def _reset_empty_state(self) -> None:
        """Reset only the in-memory state."""
        self._index = None
        self._dimension = None
        self._product_to_vector = {}
        self._vector_to_product = {}
        self._next_vector_id = 0

    @staticmethod
    def _validate_dimension(
        dimension: int,
    ) -> None:
        """Validate an embedding dimension."""
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise FAISSManagerError(
                "La dimension FAISS doit être "
                "un entier positif."
            )


# ======================================================================
# MODULE HELPERS
# ======================================================================


def _validate_product_id(
    product_id: str,
) -> str:
    """Validate and normalize a ProductID."""
    if (
        not isinstance(product_id, str)
        or not product_id.strip()
    ):
        raise FAISSManagerError(
            "product_id doit être une chaîne non vide."
        )

    return product_id.strip()


def _write_state(
    path: Path,
    state: dict[str, Any],
) -> None:
    """Write JSON state and flush it to disk."""
    try:
        with path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:
            json.dump(
                state,
                file_handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )

            file_handle.flush()
            os.fsync(
                file_handle.fileno()
            )

    except OSError as exc:
        raise FAISSManagerError(
            "Impossible d'écrire le mapping FAISS."
        ) from exc


def _read_state(
    path: Path,
) -> dict[str, Any]:
    """Read and validate the JSON state."""
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            state = json.load(
                file_handle
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise FAISSManagerError(
            "Le fichier de mapping FAISS est illisible."
        ) from exc

    if not isinstance(state, dict):
        raise FAISSManagerError(
            "Le mapping FAISS doit être un objet JSON."
        )

    return state


def _parse_state(
    state: dict[str, Any],
) -> tuple[
    dict[str, int],
    dict[int, str],
    int,
    int,
]:
    """Validate and restore both mapping directions."""
    if (
        state.get("version")
        != FAISSManager._STATE_VERSION
    ):
        raise FAISSManagerError(
            "Version du mapping FAISS incompatible."
        )

    dimension = state.get(
        "dimension"
    )

    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension <= 0
    ):
        raise FAISSManagerError(
            "Dimension du mapping FAISS invalide."
        )

    next_vector_id = state.get(
        "nextVectorId"
    )

    if (
        not isinstance(next_vector_id, int)
        or isinstance(next_vector_id, bool)
        or next_vector_id < 0
    ):
        raise FAISSManagerError(
            "Compteur VectorID invalide."
        )

    raw_product_to_vector = state.get(
        "productToVector"
    )

    raw_vector_to_product = state.get(
        "vectorToProduct"
    )

    if (
        not isinstance(
            raw_product_to_vector,
            dict,
        )
        or not isinstance(
            raw_vector_to_product,
            dict,
        )
    ):
        raise FAISSManagerError(
            "Mappings FAISS absents ou invalides."
        )

    product_to_vector: dict[str, int] = {}

    for (
        product_id,
        vector_id,
    ) in raw_product_to_vector.items():

        if (
            not isinstance(product_id, str)
            or not product_id.strip()
            or not isinstance(vector_id, int)
            or isinstance(vector_id, bool)
            or vector_id < 0
        ):
            raise FAISSManagerError(
                "Mapping ProductID -> VectorID invalide."
            )

        product_to_vector[
            product_id.strip()
        ] = vector_id

    vector_to_product: dict[int, str] = {}

    for (
        raw_vector_id,
        product_id,
    ) in raw_vector_to_product.items():

        try:
            vector_id = int(
                raw_vector_id
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise FAISSManagerError(
                "VectorID invalide dans le mapping."
            ) from exc

        if (
            vector_id < 0
            or not isinstance(
                product_id,
                str,
            )
            or not product_id.strip()
        ):
            raise FAISSManagerError(
                "Mapping VectorID -> ProductID invalide."
            )

        vector_to_product[
            vector_id
        ] = product_id.strip()

    expected_reverse = {
        vector_id: product_id
        for product_id, vector_id
        in product_to_vector.items()
    }

    if (
        vector_to_product
        != expected_reverse
    ):
        raise FAISSManagerError(
            "Les mappings ProductID/VectorID "
            "inverses ne correspondent pas."
        )

    if expected_reverse:
        maximum_id = max(
            expected_reverse
        )

        if (
            next_vector_id
            <= maximum_id
        ):
            raise FAISSManagerError(
                "Le compteur nextVectorId est incohérent."
            )

    return (
        product_to_vector,
        vector_to_product,
        next_vector_id,
        dimension,
    )


def _validate_index_mappings(
    index: faiss.IndexIDMap2,
    product_to_vector: dict[str, int],
    vector_to_product: dict[int, str],
    dimension: int,
    next_vector_id: int,
) -> None:
    """Validate persisted FAISS index against persisted mappings."""
    if int(index.d) != dimension:
        raise FAISSManagerError(
            "Dimension de l'index et du mapping incohérente."
        )

    index_ids = {
        int(vector_id)
        for vector_id in faiss.vector_to_array(
            index.id_map
        )
    }

    mapping_ids = set(
        vector_to_product
    )

    if (
        int(index.ntotal)
        != len(product_to_vector)
    ):
        raise FAISSManagerError(
            "Nombre de vecteurs FAISS et "
            "nombre de produits différents."
        )

    if index_ids != mapping_ids:
        raise FAISSManagerError(
            "VectorID FAISS et mappings persistants "
            "ne correspondent pas."
        )

    expected_reverse = {
        vector_id: product_id
        for product_id, vector_id
        in product_to_vector.items()
    }

    if (
        expected_reverse
        != vector_to_product
    ):
        raise FAISSManagerError(
            "Mappings inverses incohérents."
        )

    if mapping_ids:
        maximum_id = max(
            mapping_ids
        )

        if next_vector_id <= maximum_id:
            raise FAISSManagerError(
                "nextVectorId est inférieur ou égal "
                "au dernier VectorID utilisé."
            )


def _remove_if_present(
    path: Path,
) -> None:
    """Remove a known temporary or persistent state file."""
    try:
        path.unlink(
            missing_ok=True
        )

    except OSError as exc:
        raise FAISSManagerError(
            f"Impossible de supprimer "
            f"le fichier d'état {path.name}."
        ) from exc
