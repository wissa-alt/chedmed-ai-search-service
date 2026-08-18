"""Incremental backup synchronisation runner for the ChedMed catalogue."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import Settings, get_settings
from database.catalogue_factory import create_catalogue_client
from embeddings.embedder import EmbeddingService
from search.product_text_builder import ProductTextBuilder
from services.sync_service import SynchronizationService
from vector_store.faiss_manager import FAISSManager

LOGGER = logging.getLogger(__name__)


class BackupSyncError(RuntimeError):
    """Raised when the incremental backup synchronisation fails."""


def _read_last_sync(path: Path) -> datetime:
    """Read the last successful synchronisation timestamp.

    If no cursor exists yet, return the Unix epoch in UTC so that the
    first backup synchronisation can reconcile the complete catalogue.
    """

    if not path.exists():
        LOGGER.info(
            "Aucun curseur de synchronisation trouvé : "
            "première synchronisation incrémentale."
        )
        return datetime.fromtimestamp(0, tz=timezone.utc)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupSyncError(
            f"Impossible de lire le curseur de synchronisation : {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise BackupSyncError(
            "Le fichier last_sync.json doit contenir un objet JSON."
        )

    value = payload.get("last_sync")

    if not isinstance(value, str) or not value.strip():
        raise BackupSyncError(
            "Le champ 'last_sync' est absent ou invalide."
        )

    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise BackupSyncError(
            "Le champ 'last_sync' doit être une date ISO 8601 valide."
        ) from exc

    if timestamp.tzinfo is None:
        raise BackupSyncError(
            "Le timestamp last_sync doit contenir un fuseau horaire."
        )

    return timestamp.astimezone(timezone.utc)


def _write_last_sync(
    path: Path,
    timestamp: datetime,
) -> None:
    """Persist the last successful synchronisation timestamp atomically."""

    if timestamp.tzinfo is None:
        raise BackupSyncError(
            "Le timestamp de synchronisation doit contenir un fuseau horaire."
        )

    payload = {
        "last_sync": timestamp.astimezone(timezone.utc).isoformat(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    try:
        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(path)

    except OSError as exc:
        raise BackupSyncError(
            f"Impossible de sauvegarder le curseur : {path}"
        ) from exc


def run_backup_sync(settings: Settings) -> int:
    """Run one incremental catalogue synchronisation.

    Returns:
        0 when the synchronisation succeeds.
        1 when at least one product failed.
    """

    settings.ensure_runtime_directories()

    cursor_path = settings.last_sync_path

    last_sync = _read_last_sync(cursor_path)

    # Capture the upper bound BEFORE querying the catalogue.
    # Changes occurring during this execution will therefore be processed
    # by the next backup synchronisation instead of being lost.
    sync_started_at = datetime.now(timezone.utc)

    LOGGER.info(
        "Backup Sync démarré : last_sync=%s, upper_bound=%s",
        last_sync.isoformat(),
        sync_started_at.isoformat(),
    )

    client = create_catalogue_client(settings)
    embedder: EmbeddingService | None = None

    try:
        embedder = EmbeddingService(settings)

        vector_store = FAISSManager(settings)

        service = SynchronizationService(
            product_source=client,
            embedder=embedder,
            vector_store=vector_store,
            text_builder=ProductTextBuilder(),
        )

        report = service.sync_updated_products(last_sync)

        # Never advance the cursor when the batch contains failures.
        if report.failed_products > 0:
            LOGGER.error(
                "Backup Sync terminé avec %d échec(s). "
                "Le curseur ne sera pas avancé.",
                report.failed_products,
            )

            return 1

        _write_last_sync(
            cursor_path,
            sync_started_at,
        )

        LOGGER.info(
            "Backup Sync terminé avec succès : "
            "total=%d, ajoutés=%d, mis à jour=%d, "
            "ignorés=%d, durée=%.2fs.",
            report.total_products,
            report.indexed_products,
            report.updated_products,
            report.skipped_products,
            report.duration_seconds,
        )

        return 0

    finally:
        if embedder is not None:
            embedder.close()

        client.close()


def main() -> int:
    """Application entry point."""

    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
    )

    try:
        return run_backup_sync(settings)

    except BackupSyncError:
        LOGGER.exception(
            "Le Backup Sync a échoué."
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Erreur inattendue pendant le Backup Sync."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())