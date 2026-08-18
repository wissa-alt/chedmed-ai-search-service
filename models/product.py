
"""Product domain model returned by the ChedMed internal API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


class ProductValidationError(ValueError):
    """Raised when an API payload cannot be represented as a product."""


@dataclass(frozen=True, slots=True)
class Product:
    """A typed, immutable snapshot of a product supplied by ChedMed.

    This model is intentionally limited to API data. It contains neither
    vectors nor local persistence concerns, because ChedMed remains the
    source of truth.
    """

    id: str
    title: str
    description: str
    category: str
    brand: str | None
    color: str | None
    condition: str | None
    price: Decimal
    currency: str
    city: str | None
    image_urls: tuple[str, ...]
    status: str
    is_sold: bool
    updated_at: datetime

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Product":
        """Create a product from the ChedMed API's camelCase JSON payload.

        Args:
            payload: Raw product payload returned by the ChedMed API.

        Returns:
            A validated immutable Product instance.

        Raises:
            ProductValidationError:
                If the payload is invalid or a required field is missing.
        """
        if not isinstance(payload, Mapping):
            raise ProductValidationError(
                "Le produit reçu doit être un objet JSON."
            )

        try:
            image_urls = payload.get("imageUrls", [])

            if not isinstance(image_urls, list):
                raise ProductValidationError(
                    "imageUrls doit être une liste d'URLs texte."
                )

            if not all(
                isinstance(url, str) and url.strip()
                for url in image_urls
            ):
                raise ProductValidationError(
                    "imageUrls doit être une liste d'URLs texte."
                )

            return cls(
                id=_required_string(payload, "id"),
                title=_required_string(payload, "title"),
                description=_required_string(payload, "description"),
                category=_required_string(payload, "category"),
                brand=_optional_string(payload, "brand"),
                color=_optional_string(payload, "color"),
                condition=_optional_string(payload, "condition"),
                price=_decimal(payload, "price"),
                currency=_required_string(
                    payload,
                    "currency",
                ).upper(),
                city=_optional_string(payload, "city"),
                image_urls=tuple(
                    url.strip()
                    for url in image_urls
                ),
                status=_required_string(payload, "status"),
                is_sold=_boolean(payload, "isSold"),
                updated_at=_datetime(payload, "updatedAt"),
            )

        except KeyError as exc:
            raise ProductValidationError(
                f"Champ obligatoire absent : {exc.args[0]}."
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialise the product using ChedMed's API field naming convention."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "brand": self.brand,
            "color": self.color,
            "condition": self.condition,
            "price": str(self.price),
            "currency": self.currency,
            "city": self.city,
            "imageUrls": list(self.image_urls),
            "status": self.status,
            "isSold": self.is_sold,
            "updatedAt": self.updated_at.isoformat(),
        }


def _required_string(
    payload: Mapping[str, Any],
    field_name: str,
) -> str:
    """Return a non-empty required string field."""
    value = payload[field_name]

    if not isinstance(value, str) or not value.strip():
        raise ProductValidationError(
            f"{field_name} doit être une chaîne non vide."
        )

    return value.strip()


def _optional_string(
    payload: Mapping[str, Any],
    field_name: str,
) -> str | None:
    """Return an optional string field, normalising blanks to None."""
    value = payload.get(field_name)

    if value is None:
        return None

    if not isinstance(value, str):
        raise ProductValidationError(
            f"{field_name} doit être une chaîne ou null."
        )

    return value.strip() or None


def _decimal(
    payload: Mapping[str, Any],
    field_name: str,
) -> Decimal:
    """Parse a finite decimal price without losing precision."""
    value = payload[field_name]

    if isinstance(value, bool):
        raise ProductValidationError(
            f"{field_name} doit être un nombre valide."
        )

    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ProductValidationError(
            f"{field_name} doit être un nombre valide."
        ) from exc

    if not decimal_value.is_finite() or decimal_value < 0:
        raise ProductValidationError(
            f"{field_name} doit être un nombre positif ou nul."
        )

    return decimal_value


def _boolean(
    payload: Mapping[str, Any],
    field_name: str,
) -> bool:
    """Require an actual JSON boolean value."""
    value = payload[field_name]

    if not isinstance(value, bool):
        raise ProductValidationError(
            f"{field_name} doit être un booléen."
        )

    return value


def _datetime(
    payload: Mapping[str, Any],
    field_name: str,
) -> datetime:
    """Parse an ISO 8601 timestamp and convert it to UTC."""
    value = _required_string(
        payload,
        field_name,
    )

    try:
        timestamp = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ProductValidationError(
            f"{field_name} doit être une date ISO 8601."
        ) from exc

    if timestamp.tzinfo is None:
        raise ProductValidationError(
            f"{field_name} doit inclure un fuseau horaire."
        )

    return timestamp.astimezone(timezone.utc)
