"""Controlled errors exposed by the audio transcription boundary."""


class AudioTranscriptionError(RuntimeError):
    """Base error for an audio transcription workflow."""


class InvalidAudioError(AudioTranscriptionError, ValueError):
    """Raised when an uploaded audio file violates the public contract."""


class AudioProviderError(AudioTranscriptionError):
    """Raised when one speech-to-text provider fails or responds invalidly."""


class AudioFallbackError(AudioTranscriptionError):
    """Raised when neither the primary provider nor its fallback can succeed."""
