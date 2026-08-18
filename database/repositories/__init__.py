"""PostgreSQL repositories for typed ChedMed domain data."""

from database.repositories.product_repository import ProductRepository, ProductRepositoryError

__all__ = ["ProductRepository", "ProductRepositoryError"]
