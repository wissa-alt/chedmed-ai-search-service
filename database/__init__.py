"""Source-agnostic ChedMed catalogue access adapters."""

from database.api_client import ChedmedApiClient
from database.catalogue_client import CatalogueClient
from database.catalogue_factory import create_catalogue_client

from database.postgres_client import (
    PostgresCatalogueClient,
    PostgresCatalogueError,
    PostgresConnectionError,
    PostgresProductNotFoundError,
    PostgresQueryError,
    PostgreSQLClient,
    PostgreSQLClientError,
)
from database.repositories.product_repository import ProductNotFoundError, ProductRepository, ProductRepositoryError

__all__ = ["CatalogueClient", "ChedmedApiClient", "PostgresCatalogueClient", "PostgresCatalogueError", "PostgresConnectionError", "PostgresProductNotFoundError", "PostgresQueryError", "PostgreSQLClient", "PostgreSQLClientError", "ProductNotFoundError", "ProductRepository", "ProductRepositoryError", "create_catalogue_client"]
