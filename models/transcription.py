"""Typed, provider-neutral results for audio transcription workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TranscriptionCandidate:
    """One faithful speech-to-text hypothesis returned by a provider."""

    provider: str
    text: str
    language_hint: str | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Final transcription plus non-sensitive provenance information."""

    text: str
    primary_provider: str
    used_fallback: bool
    whisper_text: str | None = None
    gemini_text: str | None = None
    resolution_reason: str = "single_provider"
    quality_score: float = 0.0
    latency_ms: int = 0
    provider_latencies_ms: Mapping[str, int] = field(default_factory=dict)
    disagreement: Mapping[str, Any] | None = None
    alternative_text: str | None = None
    search_recovery_used: bool = False
    search_evidence: Mapping[str, Any] = field(default_factory=dict)
    gemini_triggered: bool = False
    gemini_trigger_reason: str | None = None
    whisper_search_latency_ms: int = 0
    gemini_search_latency_ms: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
