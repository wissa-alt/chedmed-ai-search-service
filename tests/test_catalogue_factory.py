"""Unit tests for configuration-driven catalogue-source selection."""

from __future__ import annotations

from pathlib import Path

from config import Settings
from database.api_client import ChedmedApiClient
from database.catalogue_factory import create_catalogue_client
from database.postgres_client import PostgresCatalogueClient


def _settings(tmp_path: Path, source: str) -> Settings:
    """Build complete settings for one source without environment access."""
    return Settings(
        environment="test", host="127.0.0.1", port=5000, log_level="CRITICAL",
        chedmed_webhook_secret="secret", catalogue_source=source,
        db_host="127.0.0.1", db_port=5432, db_name="chedmed", db_user="test", db_password="password",
        chedmed_api_base_url="https://chedmed.test", chedmed_api_token="token",
        groq_api_key="groq", project_root=tmp_path,
    )


def test_factory_selects_postgres_client(tmp_path: Path, mocker) -> None:
    """Development configuration selects PostgreSQL without touching services."""
    fake_client = mocker.Mock(spec=PostgresCatalogueClient)
    factory = mocker.patch("database.catalogue_factory.PostgresCatalogueClient", return_value=fake_client)

    assert create_catalogue_client(_settings(tmp_path, "postgres")) is fake_client
    factory.assert_called_once()


def test_factory_selects_api_client(tmp_path: Path, mocker) -> None:
    """Production configuration selects the original HTTP adapter."""
    fake_client = mocker.Mock(spec=ChedmedApiClient)
    factory = mocker.patch("database.catalogue_factory.ChedmedApiClient", return_value=fake_client)

    assert create_catalogue_client(_settings(tmp_path, "api")) is fake_client
    factory.assert_called_once()
