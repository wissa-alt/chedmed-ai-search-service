"""Audio-domain models exposed from one stable application module."""

from models.transcription import TranscriptionCandidate, TranscriptionResult

__all__ = ["TranscriptionCandidate", "TranscriptionResult"]
