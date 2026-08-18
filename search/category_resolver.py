"""Deterministic resolution of user category hints to catalogue values."""

from __future__ import annotations

from pathlib import Path
import re

from search.category_normalizer import CategoryNormalizer
from search.synonym_resources import load_synonym_groups, normalize_synonym_text


class CategoryResolver:
    """Resolve aliases without depending on the LLM, Flask, or FAISS."""

    def __init__(self, synonyms_path: str | Path | None = None) -> None:
        self._normalizer = CategoryNormalizer(synonyms_path)
        self._groups = load_synonym_groups(synonyms_path)

    def resolve(self, hint: str | None) -> str | None:
        """Return an official ChedMed category, never an unresolved hint."""
        return self._normalizer.normalize(hint)

    def resolve_query(self, text: str) -> str | None:
        """Resolve the first complete category alias present in query text."""
        normalized = normalize_synonym_text(text)
        for category, aliases in self._groups.items():
            for alias in aliases:
                candidate = normalize_synonym_text(alias)
                if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized):
                    return category
        return None

    def categories(self) -> tuple[str, ...]:
        """Expose the catalogue categories supported by the resource."""
        return self._normalizer.categories()
