"""Composition helper selecting the configured catalogue adapter."""

from __future__ import annotations

from database.api_client import ChedmedApiClient
from database.catalogue_client import CatalogueClient
from database.postgres_client import PostgresCatalogueClient
from config import Settings


def create_catalogue_client(settings: Settings) -> CatalogueClient:
    """Create the sole catalogue source selected by ``CATALOGUE_SOURCE``."""
    if settings.catalogue_source == "api":
        return ChedmedApiClient(settings)
    if settings.catalogue_source == "postgres":
        return PostgresCatalogueClient(settings)
    raise ValueError(f"Source catalogue non prise en charge : {settings.catalogue_source}.")
