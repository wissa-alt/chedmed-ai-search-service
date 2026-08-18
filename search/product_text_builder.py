"""Build deterministic semantic documents from ChedMed products."""

from __future__ import annotations

from models.product import Product


class ProductTextBuilder:
    """Build rich and deterministic semantic documents for products."""

    def build(self, product: Product) -> str:
        """Build the semantic document used by the embedding model.

        The builder intentionally does not add the ``passage:`` prefix.
        ``EmbeddingService`` is responsible for applying the E5 prefix.

        Args:
            product: Product to convert into a semantic document.

        Returns:
            A deterministic textual representation of the product.

        Raises:
            TypeError:
                If ``product`` is not a Product instance.
            ValueError:
                If the product contains no usable semantic information.
        """
        if not isinstance(product, Product):
            raise TypeError(
                "product doit être une instance de Product."
            )

        sections: list[str] = []

        self._append(sections, "Titre", product.title)
        self._append(sections, "Description", product.description)
        self._append(sections, "Catégorie", product.category)
        self._append(sections, "Marque", product.brand)
        self._append(sections, "Couleur", product.color)
        self._append(sections, "État", product.condition)
        self._append(sections, "Ville", product.city)

        if product.price is not None:
            currency = product.currency or ""
            price = f"{product.price} {currency}".strip()
            self._append(sections, "Prix", price)

        self._append(sections, "Statut", product.status)

        if not sections:
            raise ValueError(
                "Le produit ne contient aucune information "
                "exploitable pour générer un document sémantique."
            )

        return "\n".join(sections)

    @staticmethod
    def _append(
        sections: list[str],
        label: str,
        value: object | None,
    ) -> None:
        """Append a semantic field when its value is meaningful."""

        if value is None:
            return

        text = str(value).strip()

        if not text:
            return

        sections.append(f"{label}: {text}")