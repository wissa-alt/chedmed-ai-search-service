"""Centralised, typed configuration for the ChedMed AI Search service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when a required or invalid configuration value is detected."""


PROJECT_ROOT = Path(__file__).resolve().parent


def _required(name: str) -> str:
    """Return a non-empty environment variable or raise a clear error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"La variable d'environnement {name} est obligatoire.")
    return value


def _positive_int(name: str, default: int) -> int:
    """Read and validate a strictly positive integer environment variable."""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} doit être un entier.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} doit être strictement positif.")
    return value


def _boolean(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable using common textual values."""
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} doit être une valeur booléenne.")


def _non_negative_float(name: str, default: float) -> float:
    """Read a finite, non-negative floating-point setting."""
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} doit être un nombre.") from exc
    if not (value >= 0 and value < float("inf")):
        raise ConfigurationError(f"{name} doit être positif ou nul et fini.")
    return value


def _finite_float(name: str, default: float) -> float:
    """Read a finite floating-point setting."""
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} doit être un nombre.") from exc
    if value != value or abs(value) == float("inf"):
        raise ConfigurationError(f"{name} doit être fini.")
    return value


def _unit_float(name: str, default: float) -> float:
    """Read a floating-point setting constrained to [0, 1]."""
    value = _finite_float(name, default)
    if not 0 <= value <= 1:
        raise ConfigurationError(f"{name} doit être compris entre 0 et 1.")
    return value


def _catalogue_source() -> str:
    """Return the configured supported catalogue source."""
    source = os.getenv("CATALOGUE_SOURCE", "postgres").strip().lower()
    if source not in {"api", "postgres"}:
        raise ConfigurationError("CATALOGUE_SOURCE doit être api ou postgres.")
    return source


def _audio_transcription_mode() -> str:
    """Return one supported provider orchestration mode."""
    mode = os.getenv("AUDIO_TRANSCRIPTION_MODE", "fallback").strip().lower()
    if mode not in {"whisper", "gemini", "fallback", "dual"}:
        raise ConfigurationError(
            "AUDIO_TRANSCRIPTION_MODE doit être whisper, gemini, fallback ou dual."
        )
    return mode


def _mime_types(name: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated, non-empty MIME allowlist."""
    raw_value = os.getenv(name, ",".join(defaults))
    values = tuple(dict.fromkeys(item.strip().lower() for item in raw_value.split(",") if item.strip()))
    if not values or any("/" not in value for value in values):
        raise ConfigurationError(f"{name} doit contenir des types MIME séparés par des virgules.")
    return values


def _base_url(name: str) -> str:
    """Read and validate a required HTTP(S) base URL."""
    value = _required(name).rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{name} doit être une URL HTTP(S) valide.")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings, read once from environment variables."""

    environment: str
    host: str
    port: int
    log_level: str
    chedmed_webhook_secret: str = field(repr=False, default="")
    catalogue_source: str = "postgres"
    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    db_password: str = field(repr=False, default="")
    chedmed_api_base_url: str | None = None
    chedmed_api_token: str | None = field(repr=False, default=None)
    sync_page_size: int = 500
    sync_interval_minutes: int = 15
    embedding_model_name: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "cpu"
    faiss_top_k_default: int = 5
    relevance_leader_margin: float = 0.015
    relevance_max_relative_drop: float = 0.04
    relevance_min_token_length: int = 4
    groq_api_key: str = field(repr=False, default="")
    groq_chat_model: str = "openai/gpt-oss-120b"
    groq_whisper_model: str = "whisper-large-v3"
    gemini_api_key: str | None = field(repr=False, default=None)
    gemini_model: str = "gemini-2.5-flash"
    gemini_audio_model: str = "gemini-3.6-flash"
    audio_transcription_mode: str = "fallback"
    audio_max_bytes: int = 19 * 1024 * 1024
    audio_allowed_mime_types: tuple[str, ...] = (
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
        "audio/ogg",
        "audio/flac",
        "audio/x-flac",
        "audio/aiff",
        "audio/aac",
        "audio/webm",
    )
    audio_fallback_enabled: bool = True
    audio_quality_threshold: float = 0.65
    audio_max_no_speech_prob: float = 0.60
    audio_min_avg_logprob: float = -1.0
    audio_log_transcripts: bool = False
    image_search_enabled: bool = False
    gemini_image_model: str = "gemini-2.5-flash"
    image_max_bytes: int = 10 * 1024 * 1024
    image_allowed_mime_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    )
    project_root: Path = field(default=PROJECT_ROOT)

    @property
    def cache_directory(self) -> Path:
        """Directory containing durable local synchronisation state."""
        return self.project_root / "cache"

    @property
    def indexes_directory(self) -> Path:
        """Directory containing durable FAISS index files."""
        return self.project_root / "indexes"

    @property
    def logs_directory(self) -> Path:
        """Directory containing application logs."""
        return self.project_root / "logs"

    @property
    def faiss_index_path(self) -> Path:
        """Location of the persisted FAISS index."""
        return self.indexes_directory / "faiss.index"

    @property
    def id_mapping_path(self) -> Path:
        """Location of the ProductID-to-VectorID mapping."""
        return self.cache_directory / "id_mapping.json"

    @property
    def last_sync_path(self) -> Path:
        """Location of the backup-sync cursor."""
        return self.cache_directory / "last_sync.json"

    @property
    def postgres_dsn(self) -> str:
        """Return a psycopg-compatible DSN without exposing it in logs."""
        if not all((self.db_host, self.db_name, self.db_user, self.db_password)):
            raise ConfigurationError("La configuration PostgreSQL est incomplète.")
        return f"host={self.db_host} port={self.db_port} dbname={self.db_name} user={self.db_user} password={self.db_password}"

    
    def ensure_runtime_directories(self) -> None:
        """Create local runtime directories if they do not already exist."""
        for directory in (self.cache_directory, self.indexes_directory, self.logs_directory):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls, env_file: Path | None = None) -> "Settings":
        """Build validated settings from ``.env`` and process environment."""
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        catalogue_source = _catalogue_source()
        postgres_values = (
            _required("POSTGRES_HOST"),
            _positive_int("POSTGRES_PORT", 5432),
            _required("POSTGRES_DATABASE"),
            _required("POSTGRES_USER"),
            _required("POSTGRES_PASSWORD"),
        ) if catalogue_source == "postgres" else ("", 5432, "", "", "")
        return cls(
            environment=os.getenv("APP_ENV", "development").strip(),
            host=os.getenv("APP_HOST", "0.0.0.0").strip(),
            port=_positive_int("APP_PORT", 5000),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            catalogue_source=catalogue_source,
            db_host=postgres_values[0],
            db_port=postgres_values[1],
            db_name=postgres_values[2],
            db_user=postgres_values[3],
            db_password=postgres_values[4],
            chedmed_api_base_url=_base_url("CHEDMED_API_BASE_URL") if catalogue_source == "api" else None,
            chedmed_api_token=_required("CHEDMED_API_TOKEN") if catalogue_source == "api" else None,
            chedmed_webhook_secret=_required("CHEDMED_WEBHOOK_SECRET"),
            sync_page_size=_positive_int("SYNC_PAGE_SIZE", 500),
            sync_interval_minutes=_positive_int("SYNC_INTERVAL_MINUTES", 15),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base"
            ).strip(),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu").strip(),
            faiss_top_k_default=_positive_int("FAISS_TOP_K_DEFAULT", 5),
            relevance_leader_margin=_non_negative_float(
                "RELEVANCE_LEADER_MARGIN", 0.015
            ),
            relevance_max_relative_drop=_non_negative_float(
                "RELEVANCE_MAX_RELATIVE_DROP", 0.04
            ),
            relevance_min_token_length=_positive_int(
                "RELEVANCE_MIN_TOKEN_LENGTH", 4
            ),
            groq_api_key=_required("GROQ_API_KEY"),
            groq_chat_model=os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b").strip(),
            groq_whisper_model=os.getenv(
                "GROQ_WHISPER_MODEL", "whisper-large-v3"
            ).strip(),
            gemini_api_key=gemini_api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            gemini_audio_model=os.getenv(
                "GEMINI_AUDIO_MODEL", "gemini-3.6-flash"
            ).strip(),
            audio_transcription_mode=_audio_transcription_mode(),
            audio_max_bytes=_positive_int("AUDIO_MAX_BYTES", 19 * 1024 * 1024),
            audio_allowed_mime_types=_mime_types(
                "AUDIO_ALLOWED_MIME_TYPES",
                cls.__dataclass_fields__["audio_allowed_mime_types"].default,
            ),
            audio_fallback_enabled=_boolean("AUDIO_FALLBACK_ENABLED", True),
            audio_quality_threshold=_unit_float("AUDIO_QUALITY_THRESHOLD", 0.65),
            audio_max_no_speech_prob=_unit_float(
                "AUDIO_MAX_NO_SPEECH_PROB", 0.60
            ),
            audio_min_avg_logprob=_finite_float("AUDIO_MIN_AVG_LOGPROB", -1.0),
            audio_log_transcripts=_boolean("AUDIO_LOG_TRANSCRIPTS", False),
            image_search_enabled=_boolean("IMAGE_SEARCH_ENABLED"),
            gemini_image_model=os.getenv(
                "GEMINI_IMAGE_MODEL", "gemini-2.5-flash"
            ).strip(),
            image_max_bytes=_positive_int("IMAGE_MAX_BYTES", 10 * 1024 * 1024),
            image_allowed_mime_types=_mime_types(
                "IMAGE_ALLOWED_MIME_TYPES",
                cls.__dataclass_fields__["image_allowed_mime_types"].default,
            ),
        )


def get_settings() -> Settings:
    """Build and return one validated settings object for the caller.

    The composition root owns the returned instance and passes it explicitly to
    its dependencies. No process-wide configuration singleton is retained.
    """
    return Settings.from_environment()
