"""HTTP implementation of the common ChedMed catalogue client contract."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import Settings
from models.product import Product, ProductValidationError

LOGGER = logging.getLogger(__name__)


class ChedmedApiError(RuntimeError):
    """Base exception for ChedMed API catalogue failures."""


class ChedmedNetworkError(ChedmedApiError):
    """Raised when the ChedMed API cannot be reached."""


class ChedmedHttpError(ChedmedApiError):
    """Raised when the ChedMed API returns an unsuccessful status."""


class ChedmedNotFoundError(ChedmedHttpError):
    """Raised when a requested ChedMed product is absent."""


class ChedmedResponseError(ChedmedApiError):
    """Raised when ChedMed returns an invalid product payload."""


class ChedmedApiClient:
    """Read products from the future official ChedMed REST API."""

    _PRODUCTS_PATH = "/internal/ai/products"

    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        """Create one authenticated reusable HTTP session.

        Raises:
            ChedmedApiError: If API configuration is unavailable.
        """
        if not settings.chedmed_api_base_url or not settings.chedmed_api_token:
            raise ChedmedApiError("La configuration de l'API ChedMed est incomplète.")
        self._base_url = settings.chedmed_api_base_url
        self._page_size = settings.sync_page_size
        self._session = session or requests.Session()
        self._session.headers.update({"Accept": "application/json", "Authorization": f"Bearer {settings.chedmed_api_token}"})

    def close(self) -> None:
        """Close the owned HTTP session."""
        self._session.close()

    def __enter__(self) -> "ChedmedApiClient":
        """Return the open API client."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the HTTP session when leaving the context."""
        self.close()

    def get_products_page(self, page: int, limit: int) -> list[Product]:
        """Return one one-based page of products."""
        if page < 1 or limit < 1:
            raise ValueError("page et limit doivent être strictement positifs.")
        return self._to_products(self._get_json(self._PRODUCTS_PATH, {"page": page, "limit": limit}))

    def get_all_products(self) -> list[Product]:
        """Return all products by following API pages."""
        products: list[Product] = []
        page = 1
        while True:
            current = self.get_products_page(page, self._page_size)
            products.extend(current)
            if len(current) < self._page_size:
                return products
            page += 1

    def get_product(self, product_id: str) -> Product:
        """Return one product from the official source of truth."""
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id ne peut pas être vide.")
        payload = self._get_json(f"{self._PRODUCTS_PATH}/{product_id.strip()}")
        if not isinstance(payload, Mapping):
            raise ChedmedResponseError("La réponse produit doit être un objet JSON.")
        try:
            return Product.from_dict(payload)
        except ProductValidationError as exc:
            raise ChedmedResponseError("Le produit retourné par ChedMed est invalide.") from exc

    def get_products_updated_after(self, updated_after: datetime | str) -> list[Product]:
        """Return products changed after the supplied timestamp."""
        value = updated_after.isoformat() if isinstance(updated_after, datetime) else updated_after
        if not isinstance(value, str) or not value.strip():
            raise ValueError("updated_after doit être une date ISO 8601 non vide.")
        return self._to_products(self._get_json(self._PRODUCTS_PATH, {"updatedAfter": value}))

    def _get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        """Execute a transiently retried GET and decode its JSON payload."""
        retrying = Retrying(stop=stop_after_attempt(3), wait=wait_exponential(min=0.2, max=2), retry=retry_if_exception_type(ChedmedNetworkError), reraise=True)
        response = retrying(self._request, path, params)
        try:
            return response.json()
        except ValueError as exc:
            raise ChedmedResponseError("ChedMed a retourné un JSON invalide.") from exc

    def _request(self, path: str, params: Mapping[str, Any] | None) -> requests.Response:
        """Perform one GET and translate transport failures."""
        try:
            response = self._session.get(f"{self._base_url}{path}", params=params, timeout=(5, 30))
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise ChedmedNotFoundError("Produit ChedMed introuvable.") from exc
            raise ChedmedHttpError("L'API ChedMed a renvoyé une erreur HTTP.") from exc
        except requests.RequestException as exc:
            raise ChedmedNetworkError("La requête vers ChedMed a échoué.") from exc

    @staticmethod
    def _to_products(payload: Any) -> list[Product]:
        """Extract API records and map them through the existing Product model."""
        records = payload if isinstance(payload, list) else payload.get("products", payload.get("items", payload.get("data"))) if isinstance(payload, Mapping) else None
        if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
            raise ChedmedResponseError("La réponse ChedMed doit contenir une liste de produits.")
        try:
            return [Product.from_dict(record) for record in records]
        except ProductValidationError as exc:
            raise ChedmedResponseError("Un produit retourné par ChedMed est invalide.") from exc
