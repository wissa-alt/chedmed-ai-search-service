"""Validated access to the shared ChedMed synonym resource."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SynonymResourceError(ValueError):
    """Raised when the synonym resource is missing or has an invalid schema."""


def default_synonyms_path() -> Path:
    """Return the repository-owned synonym file independently of the CWD."""
    return Path(__file__).resolve().parents[1] / "resources" / "synonyms.json"


def normalize_synonym_text(value: str) -> str:
    """Normalise text for exact, accent-insensitive synonym comparisons."""
    normalized = unicodedata.normalize("NFD", value.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_accents.split())


def load_synonym_groups(
    synonyms_path: str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    """Load category aliases from the ``resources/synonyms.json`` schema.

    The resource has an optional ``_meta`` object and a mandatory ``categories``
    object mapping each official ChedMed category to its semantic aliases.

    Args:
        synonyms_path: Explicit resource location, or the repository default.

    Returns:
        Official category names mapped to validated, de-duplicated aliases.

    Raises:
        SynonymResourceError: If the file is unavailable, invalid, or ambiguous.
    """
    path = (
        default_synonyms_path()
        if synonyms_path is None
        else Path(synonyms_path).expanduser()
    )

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except OSError as exc:
        raise SynonymResourceError(
            f"Le fichier de synonymes est introuvable ou illisible : {path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise SynonymResourceError(
            f"Le fichier de synonymes contient un JSON invalide : {path}."
        ) from exc

    if not isinstance(payload, Mapping):
        raise SynonymResourceError("Le fichier de synonymes doit contenir un objet JSON.")

    raw_categories = payload.get("categories")
    if not isinstance(raw_categories, Mapping) or not raw_categories:
        raise SynonymResourceError(
            "Le fichier de synonymes doit contenir un objet non vide 'categories'."
        )

    groups: dict[str, tuple[str, ...]] = {}
    alias_owners: dict[str, str] = {}
    for raw_category, raw_aliases in raw_categories.items():
        category = _required_text(raw_category, "Une catégorie officielle")
        if not isinstance(raw_aliases, list):
            raise SynonymResourceError(
                f"Les synonymes de la catégorie {category!r} doivent être une liste."
            )

        aliases = _validated_aliases(raw_aliases, category)
        groups[category] = aliases
        for alias in aliases:
            normalized_alias = normalize_synonym_text(alias)
            previous_category = alias_owners.setdefault(normalized_alias, category)
            if previous_category != category:
                raise SynonymResourceError(
                    "Un même alias ne peut pas appartenir à plusieurs catégories : "
                    f"{alias!r}."
                )

    return groups


def _validated_aliases(raw_aliases: list[Any], category: str) -> tuple[str, ...]:
    """Validate aliases of one category while retaining their resource order."""
    aliases: list[str] = []
    seen: set[str] = set()
    for raw_alias in raw_aliases:
        alias = _required_text(raw_alias, f"Un alias de la catégorie {category!r}")
        normalized_alias = normalize_synonym_text(alias)
        if normalized_alias not in seen:
            aliases.append(alias)
            seen.add(normalized_alias)

    if not aliases:
        raise SynonymResourceError(
            f"La catégorie {category!r} doit avoir au moins un alias."
        )
    return tuple(aliases)


def _required_text(value: object, label: str) -> str:
    """Return one required non-empty textual resource value."""
    if not isinstance(value, str) or not value.strip():
        raise SynonymResourceError(f"{label} doit être une chaîne non vide.")
    return value.strip()
