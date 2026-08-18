"""Webhook event model received from the ChedMed catalogue service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class WebhookEventValidationError(ValueError):
    """Raised when a webhook payload does not respect the expected contract."""


class WebhookEventType(str, Enum):
    """Catalogue event types supported by the AI search synchronisation flow."""

    PRODUCT_CREATED = "PRODUCT_CREATED"
    PRODUCT_UPDATED = "PRODUCT_UPDATED"
    PRODUCT_DELETED = "PRODUCT_DELETED"
    PRODUCT_SOLD = "PRODUCT_SOLD"
    PRODUCT_DEACTIVATED = "PRODUCT_DEACTIVATED"
    PRODUCT_REACTIVATED = "PRODUCT_REACTIVATED"


@dataclass(frozen=True, slots=True)
class WebhookEvent:
    """Validated catalogue notification, without product details.

    Product details are deliberately absent: consumers must query the ChedMed API
    after receiving an event, preserving ChedMed as the source of truth.
    """

    event_id: str
    event_type: WebhookEventType
    product_id: str
    occurred_at: datetime

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WebhookEvent":
        """Create an event from a webhook JSON payload.

        Raises:
            WebhookEventValidationError: If the payload is absent, malformed, or
                declares an unsupported event type.
        """
        if not isinstance(payload, Mapping):
            raise WebhookEventValidationError("L'événement doit être un objet JSON.")
        try:
            event_type = WebhookEventType(_required_string(payload, "eventType"))
        except ValueError as exc:
            raise WebhookEventValidationError("eventType n'est pas pris en charge.") from exc

        return cls(
            event_id=_required_string(payload, "eventId"),
            event_type=event_type,
            product_id=_required_string(payload, "productId"),
            occurred_at=_timestamp(payload, "occurredAt"),
        )

    def to_dict(self) -> dict[str, str]:
        """Serialise the event using the webhook's camelCase naming convention."""
        return {
            "eventId": self.event_id,
            "eventType": self.event_type.value,
            "productId": self.product_id,
            "occurredAt": self.occurred_at.isoformat(),
        }


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    """Read a required non-empty string from an event payload."""
    try:
        value = payload[field_name]
    except KeyError as exc:
        raise WebhookEventValidationError(f"Champ obligatoire absent : {field_name}.") from exc
    if not isinstance(value, str) or not value.strip():
        raise WebhookEventValidationError(f"{field_name} doit être une chaîne non vide.")
    return value.strip()


def _timestamp(payload: Mapping[str, Any], field_name: str) -> datetime:
    """Parse a timezone-aware ISO 8601 event timestamp as UTC."""
    value = _required_string(payload, field_name)
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WebhookEventValidationError(f"{field_name} doit être une date ISO 8601.") from exc
    if timestamp.tzinfo is None:
        raise WebhookEventValidationError(f"{field_name} doit inclure un fuseau horaire.")
    return timestamp.astimezone(timezone.utc)
