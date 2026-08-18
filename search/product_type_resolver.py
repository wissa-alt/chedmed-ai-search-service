"""Deterministic resolution and matching of product-type vocabulary."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from search.synonym_resources import normalize_synonym_text


class ProductTypeResolverError(ValueError):
    """Raised for an invalid product-type resource or hint."""


class ProductTypeResolver:
    """Resolve multilingual product hints independently of categories."""

    def __init__(self, resource_path: str | Path | None = None) -> None:
        path = Path(resource_path) if resource_path else (
            Path(__file__).resolve().parents[1] / "resources" / "product_types.json"
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductTypeResolverError("Ressource de types produit invalide.") from exc
        groups = payload.get("product_types") if isinstance(payload, Mapping) else None
        if not isinstance(groups, Mapping) or not groups:
            raise ProductTypeResolverError("'product_types' doit être un objet non vide.")
        self._groups: dict[str, tuple[str, ...]] = {}
        self._aliases: dict[str, str] = {}
        for canonical, aliases in groups.items():
            if not isinstance(canonical, str) or not canonical.strip() or not isinstance(aliases, list):
                raise ProductTypeResolverError("Groupe de type produit invalide.")
            cleaned = tuple(alias.strip() for alias in aliases if isinstance(alias, str) and alias.strip())
            if not cleaned:
                raise ProductTypeResolverError("Un type produit doit avoir des alias.")
            self._groups[canonical.strip()] = cleaned
            self._aliases[normalize_synonym_text(canonical)] = canonical.strip()
            for alias in cleaned:
                key = normalize_synonym_text(alias)
                owner = self._aliases.setdefault(key, canonical.strip())
                if owner != canonical.strip():
                    raise ProductTypeResolverError("Alias de type produit ambigu.")

    def resolve(self, hint: str | None) -> str | None:
        """Return a canonical product type or None for an unresolved hint."""
        if hint is None:
            return None
        if not isinstance(hint, str):
            raise ProductTypeResolverError("Le type produit doit être un texte ou None.")
        return self._aliases.get(normalize_synonym_text(hint))

    def aliases_for(self, product_type: str | None) -> tuple[str, ...]:
        """Return searchable aliases for one resolved type."""
        resolved = self.resolve(product_type)
        return self._groups.get(resolved, ()) if resolved else ()

    def resolve_query(self, text: str) -> str | None:
        """Resolve a complete alias occurring anywhere in query text."""
        normalized = normalize_synonym_text(text)
        for alias, canonical in self._aliases.items():
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
                return canonical
        return None
