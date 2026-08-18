"""Catalogue source-of-truth boundary."""

from database.catalogue_client import CatalogueClient
from database.catalogue_factory import create_catalogue_client

__all__ = ["CatalogueClient", "create_catalogue_client"]
