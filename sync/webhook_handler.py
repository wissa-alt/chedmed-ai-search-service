"""Framework-independent, signed ChedMed catalogue webhook handler."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from config import Settings
from models.webhook_event import WebhookEvent, WebhookEventType, WebhookEventValidationError

LOGGER = logging.getLogger(__name__)


class SignatureError(ValueError):
    """Raised when a webhook does not carry a valid HMAC SHA-256 signature."""


class DuplicateEventError(ValueError):
    """Raised internally when an already processed event ID is received again."""


class InvalidWebhookError(ValueError):
    """Raised when a webhook body or event contract is invalid."""


class WebhookSynchronizationPort(Protocol):
    """Synchronization operations selected by catalogue event type."""

    def sync_product(self, product_id: str) -> object:
        """Synchronise a created or updated product."""

    def reactivate_product(self, product_id: str) -> object:
        """Synchronise a reactivated product."""

    def remove_product(self, product_id: str) -> object:
        """Remove a genuinely deleted product."""


@dataclass(frozen=True, slots=True)
class WebhookResult:
    """Outcome returned to the thin HTTP adapter after webhook processing."""

    accepted: bool
    duplicated: bool
    event_id: str | None
    action: str | None
    message: str


class WebhookHandler:
    """Verify, deduplicate, and delegate ChedMed webhooks without Flask.

    Processed event IDs are only persisted after the delegated synchronization
    succeeds. This lets ChedMed retry a transiently failed event safely.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        settings: Settings,
        synchronization_service: WebhookSynchronizationPort,
        signature_header: str = "X-ChedMed-Signature",
    ) -> None:
        """Configure a handler using injected synchronization and application settings."""
        if not signature_header.strip():
            raise ValueError("Le nom du header de signature ne peut pas être vide.")
        self._secret = settings.chedmed_webhook_secret.encode("utf-8")
        self._signature_header = signature_header
        self._synchronization_service = synchronization_service
        self._processed_events_path = settings.cache_directory / "processed_events.json"
        self._processed_events = self._load_processed_events()
        self._lock = threading.RLock()

    def handle(self, payload: Mapping[str, Any] | str | bytes, headers: Mapping[str, str]) -> WebhookResult:
        """Validate and process one catalogue webhook.

        Use raw ``bytes`` when available so verification uses the exact HTTP body.
        Mapping payloads are signed against deterministic compact JSON.

        Raises:
            SignatureError: If HMAC verification fails.
            InvalidWebhookError: If body JSON or event fields are invalid.
        """
        started_at = perf_counter()
        event_id: str | None = None
        LOGGER.info("Webhook ChedMed reçu.")
        try:
            if not isinstance(headers, Mapping):
                raise InvalidWebhookError("Les headers webhook doivent être une table de valeurs.")
            body, payload_dict = _normalise_payload(payload)
            self.verify_signature(body, headers)
            try:
                event = WebhookEvent.from_dict(payload_dict)
            except WebhookEventValidationError as exc:
                raise InvalidWebhookError("Le payload webhook est invalide.") from exc
            event_id = event.event_id

            with self._lock:
                try:
                    self._ensure_not_processed(event.event_id)
                except DuplicateEventError:
                    LOGGER.info("Webhook dupliqué ignoré : %s.", event.event_id)
                    return WebhookResult(
                        accepted=True,
                        duplicated=True,
                        event_id=event.event_id,
                        action="ignored",
                        message="Événement déjà traité.",
                    )

                try:
                    action = self._dispatch(event)
                    self._mark_processed(event.event_id)
                except Exception as exc:
                    LOGGER.exception("Échec du traitement du webhook %s.", event.event_id)
                    return WebhookResult(
                        accepted=False,
                        duplicated=False,
                        event_id=event.event_id,
                        action=None,
                        message=f"Échec de synchronisation : {exc}",
                    )

            LOGGER.info("Webhook %s traité (%s).", event.event_id, action)
            return WebhookResult(
                accepted=True,
                duplicated=False,
                event_id=event.event_id,
                action=action,
                message="Événement traité.",
            )
        except (SignatureError, InvalidWebhookError):
            LOGGER.warning("Webhook ChedMed rejeté%s.", f" : {event_id}" if event_id else "")
            raise
        finally:
            LOGGER.info(
                "Durée de traitement du webhook%s : %.3f s.",
                f" {event_id}" if event_id else "",
                perf_counter() - started_at,
            )

    def verify_signature(self, body: bytes, headers: Mapping[str, str]) -> None:
        """Verify a SHA-256 HMAC header using constant-time comparison.

        The accepted header value is either a lowercase/uppercase hexadecimal
        digest or the common ``sha256=<digest>`` representation.
        """
        provided_signature = _header_value(headers, self._signature_header)
        if not provided_signature:
            LOGGER.warning("Webhook rejeté : signature absente.")
            raise SignatureError("La signature webhook est absente.")
        digest = hmac.new(self._secret, body, hashlib.sha256).hexdigest()
        candidate = provided_signature.removeprefix("sha256=").strip()
        if not hmac.compare_digest(candidate.lower(), digest):
            LOGGER.warning("Webhook rejeté : signature invalide.")
            raise SignatureError("La signature webhook est invalide.")
        LOGGER.debug("Signature webhook ChedMed valide.")

    def _dispatch(self, event: WebhookEvent) -> str:
        """Delegate a validated event to the one relevant synchronization method."""
        if event.event_type in {WebhookEventType.PRODUCT_CREATED, WebhookEventType.PRODUCT_UPDATED}:
            self._synchronization_service.sync_product(event.product_id)
            return "synced"
        if event.event_type is WebhookEventType.PRODUCT_REACTIVATED:
            self._synchronization_service.reactivate_product(event.product_id)
            return "reactivated"
        if event.event_type is WebhookEventType.PRODUCT_DELETED:
            self._synchronization_service.remove_product(event.product_id)
            return "removed"
        if event.event_type in {
            WebhookEventType.PRODUCT_SOLD,
            WebhookEventType.PRODUCT_DEACTIVATED,
        }:
            # Sold/deactivated rows still exist in PostgreSQL and remain useful
            # for full-catalogue search and historical price comparisons.
            self._synchronization_service.sync_product(event.product_id)
            return "synced"
        raise InvalidWebhookError(f"Type d'événement non pris en charge : {event.event_type}.")

    def _ensure_not_processed(self, event_id: str) -> None:
        """Raise if the event ID is already present in durable handler state."""
        if event_id in self._processed_events:
            raise DuplicateEventError(f"L'événement {event_id} a déjà été traité.")

    def _mark_processed(self, event_id: str) -> None:
        """Persist a successfully handled event ID with a future-TTL-ready timestamp."""
        self._processed_events[event_id] = datetime.now(timezone.utc).isoformat()
        self._save_processed_events()

    def _load_processed_events(self) -> dict[str, str]:
        """Restore validated processed-event state, or start empty when absent."""
        if not self._processed_events_path.exists():
            return {}
        try:
            with self._processed_events_path.open("r", encoding="utf-8") as file_handle:
                state = json.load(file_handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidWebhookError("Le cache des événements traités est illisible.") from exc
        if (
            not isinstance(state, dict)
            or state.get("version") != self._STATE_VERSION
            or not isinstance(state.get("events"), dict)
            or not all(
                isinstance(event_id, str)
                and event_id.strip()
                and isinstance(processed_at, str)
                for event_id, processed_at in state["events"].items()
            )
        ):
            raise InvalidWebhookError("Le cache des événements traités est invalide.")
        return dict(state["events"])

    def _save_processed_events(self) -> None:
        """Persist event IDs via an atomic replacement to survive process crashes."""
        self._processed_events_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._processed_events_path.with_suffix(
            f"{self._processed_events_path.suffix}.tmp"
        )
        state = {"version": self._STATE_VERSION, "events": self._processed_events}
        try:
            with temporary_path.open("w", encoding="utf-8") as file_handle:
                json.dump(state, file_handle, ensure_ascii=False, sort_keys=True, indent=2)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            os.replace(temporary_path, self._processed_events_path)
        except OSError as exc:
            temporary_path.unlink(missing_ok=True)
            raise InvalidWebhookError("Impossible de sauvegarder les événements traités.") from exc


def _normalise_payload(payload: Mapping[str, Any] | str | bytes) -> tuple[bytes, Mapping[str, Any]]:
    """Decode a webhook payload while preserving raw body bytes for HMAC checks."""
    if isinstance(payload, bytes):
        body = payload
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidWebhookError("Le corps webhook doit être du JSON UTF-8 valide.") from exc
    elif isinstance(payload, str):
        body = payload.encode("utf-8")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise InvalidWebhookError("Le corps webhook doit être du JSON valide.") from exc
    elif isinstance(payload, Mapping):
        decoded = dict(payload)
        try:
            body = json.dumps(
                decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise InvalidWebhookError("Le payload webhook ne peut pas être sérialisé.") from exc
    else:
        raise InvalidWebhookError("Le payload webhook doit être un objet JSON ou un corps JSON.")
    if not isinstance(decoded, Mapping):
        raise InvalidWebhookError("Le payload webhook doit être un objet JSON.")
    return body, decoded


def _header_value(headers: Mapping[str, str], target_name: str) -> str | None:
    """Find a header case-insensitively for plain mappings and Flask adapters."""
    for name, value in headers.items():
        if name.lower() == target_name.lower() and isinstance(value, str):
            return value
    return None
