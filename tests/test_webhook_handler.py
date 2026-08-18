"""Unit tests for signed, deduplicated ChedMed webhook processing."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from config import Settings
from sync.webhook_handler import InvalidWebhookError, SignatureError, WebhookHandler


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Provide an isolated processed-event cache and known HMAC secret."""
    return Settings(
        environment="test",
        host="127.0.0.1",
        port=5000,
        log_level="CRITICAL",
        db_host="127.0.0.1", db_port=5432, db_name="chedmed", db_user="test", db_password="password",
        chedmed_webhook_secret="webhook-test-secret",
        groq_api_key="test-groq-key",
        project_root=tmp_path,
    )


@pytest.fixture
def synchronization_service() -> Mock:
    """Return an injected synchronization-service double."""
    return Mock()


@pytest.fixture
def handler(settings: Settings, synchronization_service: Mock) -> WebhookHandler:
    """Create a framework-independent handler."""
    return WebhookHandler(settings, synchronization_service)


def _payload(event_type: str, event_id: str = "event-1") -> dict[str, str]:
    """Return a complete webhook event payload."""
    return {
        "eventId": event_id,
        "eventType": event_type,
        "productId": "product-42",
        "occurredAt": "2026-07-28T12:30:00Z",
    }


def _signed_headers(settings: Settings, payload: dict[str, str]) -> dict[str, str]:
    """Sign the deterministic JSON representation used for mapping payloads."""
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    signature = hmac.new(
        settings.chedmed_webhook_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return {"X-ChedMed-Signature": f"sha256={signature}"}


@pytest.mark.parametrize("event_type", ["PRODUCT_CREATED", "PRODUCT_UPDATED"])
def test_created_and_updated_events_sync_product(
    handler: WebhookHandler,
    settings: Settings,
    synchronization_service: Mock,
    event_type: str,
) -> None:
    """Creation and update delegate only to targeted product synchronization."""
    payload = _payload(event_type)

    result = handler.handle(payload, _signed_headers(settings, payload))

    assert result.accepted is True
    assert result.action == "synced"
    synchronization_service.sync_product.assert_called_once_with("product-42")


def test_deleted_event_delegates_to_remove_product(
    handler: WebhookHandler,
    settings: Settings,
    synchronization_service: Mock,
) -> None:
    """Only a genuine deletion removes a product from the vector store."""
    payload = _payload("PRODUCT_DELETED")

    result = handler.handle(payload, _signed_headers(settings, payload))

    assert result.accepted is True
    assert result.action == "removed"
    synchronization_service.remove_product.assert_called_once_with("product-42")


@pytest.mark.parametrize("event_type", ["PRODUCT_SOLD", "PRODUCT_DEACTIVATED"])
def test_non_deleted_status_events_refresh_product_instead_of_removing_it(
    handler: WebhookHandler,
    settings: Settings,
    synchronization_service: Mock,
    event_type: str,
) -> None:
    payload = _payload(event_type)

    result = handler.handle(payload, _signed_headers(settings, payload))

    assert result.accepted is True
    assert result.action == "synced"
    synchronization_service.sync_product.assert_called_once_with("product-42")
    synchronization_service.remove_product.assert_not_called()


def test_reactivated_event_delegates_to_reactivation(
    handler: WebhookHandler, settings: Settings, synchronization_service: Mock
) -> None:
    """Reactivation refreshes the product from ChedMed through the service."""
    payload = _payload("PRODUCT_REACTIVATED")

    result = handler.handle(payload, _signed_headers(settings, payload))

    assert result.accepted is True
    assert result.action == "reactivated"
    synchronization_service.reactivate_product.assert_called_once_with("product-42")


def test_invalid_signature_rejects_before_dispatch(
    handler: WebhookHandler, synchronization_service: Mock
) -> None:
    """A bad signature neither dispatches an action nor marks an event processed."""
    with pytest.raises(SignatureError):
        handler.handle(_payload("PRODUCT_CREATED"), {"X-ChedMed-Signature": "invalid"})

    synchronization_service.sync_product.assert_not_called()


def test_duplicate_event_is_ignored_after_successful_processing(
    handler: WebhookHandler, settings: Settings, synchronization_service: Mock
) -> None:
    """The same event ID is safely acknowledged once without a second delegation."""
    payload = _payload("PRODUCT_CREATED", event_id="event-duplicate")
    headers = _signed_headers(settings, payload)

    first_result = handler.handle(payload, headers)
    duplicate_result = handler.handle(payload, headers)

    assert first_result.accepted is True
    assert duplicate_result.accepted is True
    assert duplicate_result.duplicated is True
    assert duplicate_result.action == "ignored"
    synchronization_service.sync_product.assert_called_once()


def test_invalid_payload_is_rejected_after_signature_validation(
    handler: WebhookHandler, settings: Settings
) -> None:
    """Malformed event fields are translated to the handler's explicit error."""
    payload = {"eventId": "event-invalid", "eventType": "PRODUCT_CREATED"}
    headers = _signed_headers(settings, payload)

    with pytest.raises(InvalidWebhookError):
        handler.handle(payload, headers)


def test_unknown_event_type_is_rejected(
    handler: WebhookHandler, settings: Settings
) -> None:
    """Only the six documented catalogue event types are accepted."""
    payload = _payload("PRODUCT_UNKNOWN")

    with pytest.raises(InvalidWebhookError):
        handler.handle(payload, _signed_headers(settings, payload))
