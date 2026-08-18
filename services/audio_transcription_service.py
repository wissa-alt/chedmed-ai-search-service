"""Provider-neutral orchestration for validated, in-memory audio transcription."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Protocol

from models.transcription import TranscriptionCandidate, TranscriptionResult
from services.audio_errors import AudioFallbackError, AudioProviderError, InvalidAudioError
from services.transcription_quality import (
    ResolutionDecision,
    TranscriptionQuality,
    TranscriptionQualityEvaluator,
    TranscriptionResolver,
)

LOGGER = logging.getLogger(__name__)
VALID_MODES = frozenset({"whisper", "gemini", "fallback", "dual"})


class TranscriptionProvider(Protocol):
    """Transcribe the original audio bytes into one provider hypothesis."""

    name: str

    def transcribe(
        self, audio: bytes, filename: str, content_type: str
    ) -> TranscriptionCandidate:
        """Return a typed candidate or raise ``AudioProviderError``."""


@dataclass(frozen=True, slots=True)
class AudioValidationPolicy:
    """Public upload limits shared by HTTP and non-HTTP callers."""

    max_bytes: int
    allowed_mime_types: frozenset[str]


class AudioTranscriptionService:
    """Validate audio, execute configured providers, and select faithful text."""

    def __init__(
        self,
        whisper_provider: TranscriptionProvider | None,
        gemini_provider: TranscriptionProvider | None,
        *,
        mode: str = "fallback",
        validation_policy: AudioValidationPolicy,
        evaluator: TranscriptionQualityEvaluator | None = None,
        resolver: TranscriptionResolver | None = None,
        log_transcripts: bool = False,
    ) -> None:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in VALID_MODES:
            raise ValueError(f"Mode de transcription audio invalide: {mode}")
        self._whisper = whisper_provider
        self._gemini = gemini_provider
        self._mode = normalized_mode
        self._policy = validation_policy
        self._evaluator = evaluator or TranscriptionQualityEvaluator()
        self._resolver = resolver or TranscriptionResolver(self._evaluator)
        self._log_transcripts = log_transcripts

    def transcribe(
        self, audio: bytes, filename: str, content_type: str
    ) -> TranscriptionResult:
        """Return a final faithful transcript according to the configured mode."""
        mime_type = self._validate(audio, filename, content_type)
        LOGGER.info(
            "Transcription audio: filename=%s mime=%s bytes=%d mode=%s",
            filename,
            mime_type,
            len(audio),
            self._mode,
        )
        started = time.monotonic()
        if self._mode == "whisper":
            result = self._single(self._whisper, audio, filename, mime_type)
        elif self._mode == "gemini":
            result = self._single(self._gemini, audio, filename, mime_type)
        elif self._mode == "dual":
            result = self._dual(audio, filename, mime_type)
        else:
            result = self._fallback(audio, filename, mime_type)
        result = replace(result, latency_ms=_elapsed_ms(started))
        LOGGER.info(
            "Transcription terminée: provider=%s fallback=%s chars=%d quality=%.3f "
            "reason=%s provider_latencies_ms=%s total_latency_ms=%d",
            result.primary_provider,
            result.used_fallback,
            len(result.text),
            result.quality_score,
            result.resolution_reason,
            dict(result.provider_latencies_ms),
            result.latency_ms,
        )
        if self._log_transcripts:
            LOGGER.debug("Texte final de transcription: %s", result.text)
        return result

    def transcribe_alternative(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        *,
        provider: str = "gemini",
    ) -> TranscriptionResult:
        """Transcribe the same original audio with one independent provider.

        This method exists for downstream recovery after a completed search. It
        does not compare, merge, translate, or interpret provider hypotheses.
        """
        if provider != "gemini":
            raise ValueError("Seul le provider alternatif Gemini est pris en charge.")
        mime_type = self._validate(audio, filename, content_type)
        started = time.monotonic()
        result = self._review_gemini(
            audio, filename, mime_type, whisper_text=None
        )
        result = replace(result, latency_ms=_elapsed_ms(started))
        LOGGER.info(
            "Transcription alternative terminée: provider=%s chars=%d latency_ms=%d",
            result.primary_provider,
            len(result.text),
            result.latency_ms,
        )
        return result

    def transcribe_whisper_primary(
        self, audio: bytes, filename: str, content_type: str
    ) -> TranscriptionResult:
        """Return Whisper first, including a low-quality assessment for recovery."""
        mime_type = self._validate(audio, filename, content_type)
        started = time.monotonic()
        candidate, latency = self._call(
            self._whisper, audio, filename, mime_type
        )
        quality = self._evaluate(candidate)
        result = self._result(
            candidate,
            False,
            "whisper_accepted" if quality.is_reliable else "whisper_low_quality",
            quality.quality_score,
            {"whisper": latency},
            whisper=candidate,
            quality_reasons=quality.reasons,
        )
        return replace(
            result,
            latency_ms=_elapsed_ms(started),
            metadata={
                **result.metadata,
                "is_reliable": quality.is_reliable,
            },
        )

    def review_with_gemini(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        whisper_text: str | None,
    ) -> TranscriptionResult:
        """Normalize the spoken marketplace query using the same original audio."""
        mime_type = self._validate(audio, filename, content_type)
        started = time.monotonic()
        result = self._review_gemini(
            audio,
            filename,
            mime_type,
            whisper_text=whisper_text,
            semantic_normalization=True,
        )
        return replace(result, latency_ms=_elapsed_ms(started))

    def _validate(self, audio: bytes, filename: str, content_type: str) -> str:
        if not isinstance(audio, bytes) or not audio:
            raise InvalidAudioError("Le fichier audio ne peut pas être vide.")
        if len(audio) > self._policy.max_bytes:
            raise InvalidAudioError(
                f"Le fichier audio dépasse la limite de {self._policy.max_bytes} octets."
            )
        if not isinstance(filename, str) or not filename.strip():
            raise InvalidAudioError("Le nom du fichier audio est obligatoire.")
        if not isinstance(content_type, str) or not content_type.strip():
            raise InvalidAudioError("Le type MIME audio est obligatoire.")
        mime_type = content_type.partition(";")[0].strip().lower()
        if mime_type not in self._policy.allowed_mime_types:
            raise InvalidAudioError(f"Type MIME audio non pris en charge: {mime_type}.")
        return mime_type

    def _single(
        self,
        provider: TranscriptionProvider | None,
        audio: bytes,
        filename: str,
        content_type: str,
    ) -> TranscriptionResult:
        candidate, latency = self._call(provider, audio, filename, content_type)
        quality = self._evaluate(candidate)
        if not quality.is_reliable:
            raise AudioFallbackError(
                f"La transcription {candidate.provider} est de faible qualité: "
                f"{','.join(quality.reasons) or 'quality_below_threshold'}."
            )
        return self._result(
            candidate,
            False,
            f"{candidate.provider}_accepted",
            quality.quality_score,
            {candidate.provider: latency},
            quality_reasons=quality.reasons,
        )

    def _fallback(
        self, audio: bytes, filename: str, content_type: str
    ) -> TranscriptionResult:
        whisper: TranscriptionCandidate | None = None
        whisper_quality: TranscriptionQuality | None = None
        latencies: dict[str, int] = {}
        fallback_reason = "whisper_failed"
        try:
            whisper, latencies["whisper"] = self._call(
                self._whisper, audio, filename, content_type
            )
            whisper_quality = self._evaluate(whisper)
            if whisper_quality.is_reliable:
                return self._result(
                    whisper,
                    False,
                    "whisper_accepted",
                    whisper_quality.quality_score,
                    latencies,
                    whisper=whisper,
                    quality_reasons=whisper_quality.reasons,
                )
            fallback_reason = "whisper_low_quality:" + (
                ",".join(whisper_quality.reasons) or "quality_below_threshold"
            )
        except AudioProviderError as exc:
            LOGGER.warning("Whisper a échoué; fallback Gemini: %s", exc)

        LOGGER.info("Fallback Gemini déclenché: reason=%s", fallback_reason)
        try:
            gemini, latencies["gemini"] = self._call_gemini_review(
                audio, filename, content_type, whisper.text if whisper else None
            )
        except AudioProviderError as exc:
            raise AudioFallbackError(
                "Aucun provider n'a produit une transcription fiable."
            ) from exc
        decision, resolver_latency = self._resolve(whisper, gemini)
        latencies["resolver"] = resolver_latency
        return self._decision_result(
            decision, True, latencies, whisper=whisper, gemini=gemini
        )

    def _dual(
        self, audio: bytes, filename: str, content_type: str
    ) -> TranscriptionResult:
        latencies: dict[str, int] = {}
        whisper, whisper_error, whisper_latency = self._try_call(
            self._whisper, audio, filename, content_type
        )
        latencies["whisper"] = whisper_latency
        gemini, gemini_error, gemini_latency = self._try_call(
            self._gemini, audio, filename, content_type
        )
        latencies["gemini"] = gemini_latency
        decision, resolver_latency = self._resolve(whisper, gemini)
        latencies["resolver"] = resolver_latency
        if decision.candidate is None:
            raise AudioFallbackError(
                "Whisper et Gemini n'ont pas produit de transcription fiable."
            ) from (gemini_error or whisper_error)
        return self._decision_result(
            decision,
            whisper is None and gemini is not None,
            latencies,
            whisper=whisper,
            gemini=gemini,
        )

    def _call(
        self,
        provider: TranscriptionProvider | None,
        audio: bytes,
        filename: str,
        content_type: str,
    ) -> tuple[TranscriptionCandidate, int]:
        if provider is None:
            raise AudioProviderError("Le fournisseur audio configuré est indisponible.")
        started = time.monotonic()
        try:
            candidate = provider.transcribe(audio, filename, content_type)
        except AudioProviderError:
            raise
        except Exception as exc:
            raise AudioProviderError(f"Le fournisseur {provider.name} a échoué.") from exc
        latency = _elapsed_ms(started)
        LOGGER.info(
            "Provider audio terminé: provider=%s latency_ms=%d chars=%d",
            provider.name,
            latency,
            len(candidate.text),
        )
        if candidate.provider == "whisper":
            LOGGER.info(
                "Whisper metadata: language=%s duration=%s segments=%s "
                "avg_logprob=%s max_no_speech_prob=%s",
                candidate.language_hint,
                candidate.metadata.get("duration"),
                candidate.metadata.get("segment_count"),
                candidate.metadata.get("avg_logprob"),
                candidate.metadata.get("max_no_speech_prob"),
            )
        if self._log_transcripts:
            LOGGER.debug("Transcript %s: %s", provider.name, candidate.text)
        return candidate, latency

    def _try_call(
        self,
        provider: TranscriptionProvider | None,
        audio: bytes,
        filename: str,
        content_type: str,
    ) -> tuple[TranscriptionCandidate | None, Exception | None, int]:
        started = time.monotonic()
        try:
            candidate, latency = self._call(provider, audio, filename, content_type)
            return candidate, None, latency
        except AudioProviderError as exc:
            latency = _elapsed_ms(started)
            LOGGER.warning("Provider audio indisponible en mode dual: %s", exc)
            return None, exc, latency

    def _call_gemini_review(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        whisper_text: str | None,
        *,
        semantic_normalization: bool = False,
    ) -> tuple[TranscriptionCandidate, int]:
        if self._gemini is None:
            raise AudioProviderError("Le fournisseur audio Gemini est indisponible.")
        started = time.monotonic()
        try:
            normalize = getattr(self._gemini, "normalize_search", None)
            if semantic_normalization and callable(normalize):
                candidate = normalize(
                    audio,
                    filename,
                    content_type,
                    whisper_text=whisper_text,
                )
            else:
                candidate = self._gemini.transcribe(  # type: ignore[call-arg]
                    audio,
                    filename,
                    content_type,
                    whisper_text=whisper_text,
                )
        except AudioProviderError:
            raise
        except Exception as exc:
            raise AudioProviderError("Le fournisseur gemini a échoué.") from exc
        latency = _elapsed_ms(started)
        LOGGER.info(
            "Review Gemini terminé: latency_ms=%d chars=%d whisper_context=%s",
            latency,
            len(candidate.text),
            bool(whisper_text),
        )
        return candidate, latency

    def _review_gemini(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        *,
        whisper_text: str | None,
        semantic_normalization: bool = False,
    ) -> TranscriptionResult:
        candidate, latency = self._call_gemini_review(
            audio,
            filename,
            content_type,
            whisper_text,
            semantic_normalization=semantic_normalization,
        )
        quality = self._evaluate(candidate)
        if not quality.is_reliable:
            raise AudioFallbackError("La correction Gemini est de faible qualité.")
        return self._result(
            candidate,
            True,
            "gemini_review_completed",
            quality.quality_score,
            {"gemini": latency},
            gemini=candidate,
            quality_reasons=quality.reasons,
        )

    def _evaluate(self, candidate: TranscriptionCandidate) -> TranscriptionQuality:
        quality = self._evaluator.evaluate(candidate)
        LOGGER.info(
            "Qualité transcription: provider=%s score=%.3f reliable=%s reasons=%s",
            candidate.provider,
            quality.quality_score,
            quality.is_reliable,
            quality.reasons,
        )
        return quality

    def _resolve(
        self,
        whisper: TranscriptionCandidate | None,
        gemini: TranscriptionCandidate | None,
    ) -> tuple[ResolutionDecision, int]:
        started = time.monotonic()
        decision = self._resolver.resolve(whisper, gemini)
        latency = _elapsed_ms(started)
        LOGGER.info(
            "Résolution transcription: reason=%s provider=%s quality=%.3f "
            "disagreement=%s latency_ms=%d",
            decision.reason,
            decision.candidate.provider if decision.candidate else None,
            decision.quality_score,
            decision.disagreement,
            latency,
        )
        return decision, latency

    def _decision_result(
        self,
        decision: ResolutionDecision,
        used_fallback: bool,
        latencies: dict[str, int],
        *,
        whisper: TranscriptionCandidate | None,
        gemini: TranscriptionCandidate | None,
    ) -> TranscriptionResult:
        if decision.candidate is None:
            raise AudioFallbackError("Aucune transcription fiable n'a été produite.")
        result = self._result(
            decision.candidate,
            used_fallback,
            decision.reason,
            decision.quality_score,
            latencies,
            whisper,
            gemini,
            decision.disagreement,
            decision.quality_reasons,
        )
        candidate_scores = {
            candidate.provider: self._evaluator.evaluate(candidate).quality_score
            for candidate in (whisper, gemini)
            if candidate is not None
        }
        return replace(
            result,
            metadata={
                **result.metadata,
                "candidate_quality_scores": candidate_scores,
            },
        )

    @staticmethod
    def _result(
        chosen: TranscriptionCandidate,
        used_fallback: bool,
        reason: str,
        quality_score: float,
        latencies: dict[str, int],
        whisper: TranscriptionCandidate | None = None,
        gemini: TranscriptionCandidate | None = None,
        disagreement: dict[str, object] | None = None,
        quality_reasons: tuple[str, ...] = (),
    ) -> TranscriptionResult:
        if chosen.provider == "whisper" and whisper is None:
            whisper = chosen
        if chosen.provider == "gemini" and gemini is None:
            gemini = chosen
        return TranscriptionResult(
            text=chosen.text,
            primary_provider=chosen.provider,
            used_fallback=used_fallback,
            whisper_text=whisper.text if whisper else None,
            gemini_text=gemini.text if gemini else None,
            resolution_reason=reason,
            quality_score=quality_score,
            provider_latencies_ms=dict(latencies),
            disagreement=disagreement,
            metadata={
                "language": chosen.language_hint,
                "duration": chosen.metadata.get("duration"),
                "quality_reasons": quality_reasons,
                "whisper_metadata": _safe_whisper_metadata(whisper),
            },
        )


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _safe_whisper_metadata(
    whisper: TranscriptionCandidate | None,
) -> dict[str, object]:
    if whisper is None:
        return {}
    fields = (
        "duration",
        "segment_count",
        "avg_logprob",
        "max_no_speech_prob",
        "model",
        "task",
    )
    metadata = {
        field: whisper.metadata[field]
        for field in fields
        if whisper.metadata.get(field) is not None
    }
    if whisper.language_hint:
        metadata["language"] = whisper.language_hint
    return metadata
