"""Small, deterministic product families used only for marketplace fallback."""

from __future__ import annotations

import re
import unicodedata

from models.product import Product
from models.search_query import StructuredSearchQuery


# This is deliberately a compact commercial taxonomy, not a catalogue of every
# possible product word. New families can be added without changing search
# orchestration; unknown terms simply remain handled by E5/FAISS primary search.
_FAMILIES: dict[str, tuple[str, ...]] = {
    "tops": (
        "t-shirt", "tee shirt", "shirt", "polo", "hoodie",
        "sweatshirt", "sweater", "pull", "chemise",
    ),
    "footwear": (
        "chaussures", "chaussure", "shoes", "shoe", "sneakers",
        "sneaker", "baskets", "sandals", "sandales", "chappal",
        "boots", "bottes",
    ),
    "computers": (
        "laptop", "notebook", "ultrabook", "desktop",
        "ordinateur portable", "ordinateur de bureau",
    ),
    "fragrance": (
        "parfum", "perfume", "perfumes", "eau de parfum",
        "eau de toilette", "fragrance",
    ),
}

# A deliberately small outer ring. It only supports a final marketplace
# fallback and never participates in primary retrieval or structured filters.
_BROAD_FAMILIES: dict[str, tuple[str, ...]] = {
    "tops": ("jacket", "cardigan", "coat", "veste", "blouson"),
    "footwear": ("slipper", "slippers", "chaussettes", "socks"),
    "computers": (
        "laptop bag", "laptop messenger bag", "computer bag", "keyboard", "mouse", "monitor",
        "tablet", "tablette", "gpu", "graphics card", "carte graphique", "rtx",
    ),
    "fragrance": ("body spray", "deodorant", "brume"),
}


class ProductFamilies:
    """Resolve a query and catalogue text to one of a few broad families."""

    def __init__(self) -> None:
        self._families = {
            family: tuple(_normalize(alias) for alias in aliases)
            for family, aliases in _FAMILIES.items()
        }
        self._broad_families = {
            family: tuple(_normalize(alias) for alias in aliases)
            for family, aliases in _BROAD_FAMILIES.items()
        }

    def for_query(self, query: StructuredSearchQuery) -> str | None:
        """Use reliable resolved type plus original user wording only."""
        sources = tuple(
            value for value in (query.product_type, query.original_query) if value
        )
        return self._resolve(" ".join(sources))

    def for_product(self, product: Product) -> str | None:
        """Classify from source-of-truth catalogue text, never LLM output."""
        return self._resolve(
            " ".join(
                value
                for value in (product.title, product.description, product.brand)
                if value
            )
        )

    def matches(self, product: Product, family: str) -> bool:
        # Adjacent accessories can contain the core word ("laptop bag") but
        # must remain broad rather than being promoted to the main family.
        return not self.broad_matches(product, family) and self.for_product(product) == family

    def broad_matches(self, product: Product, family: str) -> bool:
        """Return a controlled adjacent-domain match, never an arbitrary FAISS hit."""
        text = _normalize(" ".join(
            value for value in (product.title, product.description, product.brand) if value
        ))
        return any(
            _contains_phrase(text, alias)
            for alias in self._broad_families.get(family, ())
        )

    def _resolve(self, value: str) -> str | None:
        normalized = _normalize(value)
        for family, aliases in self._families.items():
            if any(_contains_phrase(normalized, alias) for alias in aliases):
                return family
        return None


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    without_accents = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(re.findall(r"[^\W_]+", without_accents, flags=re.UNICODE))


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))
