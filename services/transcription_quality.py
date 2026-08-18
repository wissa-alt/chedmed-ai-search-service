"""Explainable transcription quality scoring and conservative resolution."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from models.transcription import TranscriptionCandidate


@dataclass(frozen=True, slots=True)
class TranscriptionQuality:
    """Internal quality assessment; the score is not provider confidence."""

    quality_score: float
    is_reliable: bool
    reasons: tuple[str, ...]
    fallback_recommended: bool
    valid_character_ratio: float

    @property
    def acceptable(self) -> bool:
        """Backward-compatible alias for callers interested in reliability."""
        return self.is_reliable


@dataclass(frozen=True, slots=True)
class ResolutionDecision:
    """A selection between existing hypotheses, never a generated third text."""

    candidate: TranscriptionCandidate | None
    reason: str
    quality_score: float
    quality_reasons: tuple[str, ...] = ()
    disagreement: dict[str, Any] | None = None


class TranscriptionQualityEvaluator:
    """Combine real Whisper metadata and language-neutral text integrity checks."""

    def __init__(
        self,
        *,
        quality_threshold: float = 0.65,
        max_no_speech_prob: float = 0.60,
        min_avg_logprob: float = -1.0,
    ) -> None:
        if not 0 <= quality_threshold <= 1:
            raise ValueError("quality_threshold doit être compris entre 0 et 1.")
        if not 0 <= max_no_speech_prob <= 1:
            raise ValueError("max_no_speech_prob doit être compris entre 0 et 1.")
        self._threshold = quality_threshold
        self._max_no_speech_prob = max_no_speech_prob
        self._min_avg_logprob = min_avg_logprob

    def evaluate(self, candidate: TranscriptionCandidate) -> TranscriptionQuality:
        text = candidate.text.strip()
        reasons: list[str] = []
        score = 1.0
        if not text:
            return TranscriptionQuality(0.0, False, ("empty",), True, 0.0)

        meaningful = [char for char in text if not char.isspace()]
        valid = sum(_is_valid_character(char) for char in meaningful)
        valid_ratio = valid / len(meaningful) if meaningful else 0.0
        alphanumeric = sum(char.isalnum() for char in text)
        if alphanumeric < 2:
            reasons.append("too_short")
            score -= 0.45
        if "\ufffd" in text or any(unicodedata.category(char) == "Cc" for char in text):
            reasons.append("corrupted_characters")
            score -= 0.55
        if valid_ratio < 0.70:
            reasons.append("low_valid_character_ratio")
            score -= min(0.50, 0.70 - valid_ratio)

        tokens = re.findall(r"[\w']+", text.casefold(), flags=re.UNICODE)
        if len(tokens) >= 4 and _has_abnormal_repetition(tokens):
            reasons.append("abnormal_repetition")
            score -= 0.40
        if any(len(token) > 50 or _has_repeated_character(token) for token in tokens):
            reasons.append("aberrant_token")
            score -= 0.30

        avg_logprob = _number(candidate.metadata.get("avg_logprob"))
        if avg_logprob is not None and avg_logprob < self._min_avg_logprob:
            reasons.append("low_avg_logprob")
            score -= min(0.45, 0.25 + (self._min_avg_logprob - avg_logprob) * 0.20)
        no_speech_prob = _number(candidate.metadata.get("max_no_speech_prob"))
        if no_speech_prob is not None and no_speech_prob > self._max_no_speech_prob:
            reasons.append("high_no_speech_prob")
            score -= min(0.50, 0.30 + (no_speech_prob - self._max_no_speech_prob))

        segments = candidate.metadata.get("segments")
        if isinstance(segments, (tuple, list)) and segments:
            nonempty_segments = sum(
                bool(segment.get("text", "").strip())
                for segment in segments
                if isinstance(segment, dict)
            )
            if nonempty_segments == 0:
                reasons.append("empty_segments")
                score -= 0.50

        score = round(max(0.0, min(1.0, score)), 3)
        fatal = bool({"empty", "corrupted_characters", "empty_segments"} & set(reasons))
        reliable = score >= self._threshold and not fatal
        return TranscriptionQuality(score, reliable, tuple(reasons), not reliable, valid_ratio)


class TranscriptionResolver:
    """Choose a whole provider hypothesis without translation or token fusion."""

    def __init__(self, evaluator: TranscriptionQualityEvaluator | None = None) -> None:
        self._evaluator = evaluator or TranscriptionQualityEvaluator()

    def resolve(
        self,
        whisper: TranscriptionCandidate | None,
        gemini: TranscriptionCandidate | None,
    ) -> ResolutionDecision:
        if whisper is None and gemini is None:
            return ResolutionDecision(None, "both_invalid", 0.0)
        if whisper is None:
            quality = self._evaluator.evaluate(gemini)  # type: ignore[arg-type]
            return ResolutionDecision(
                gemini if quality.is_reliable else None,
                "whisper_failed_gemini_selected" if quality.is_reliable else "both_invalid",
                quality.quality_score,
                quality.reasons,
            )
        whisper_quality = self._evaluator.evaluate(whisper)
        if gemini is None:
            return ResolutionDecision(
                whisper if whisper_quality.is_reliable else None,
                "gemini_failed_whisper_preserved" if whisper_quality.is_reliable else "both_invalid",
                whisper_quality.quality_score,
                whisper_quality.reasons,
            )

        gemini_quality = self._evaluator.evaluate(gemini)
        similarity = _similarity(whisper.text, gemini.text)
        disagreement = {
            "similarity": round(similarity, 3),
            "strong": similarity < 0.55,
        }
        if not whisper_quality.is_reliable and not gemini_quality.is_reliable:
            return ResolutionDecision(None, "both_invalid", max(
                whisper_quality.quality_score, gemini_quality.quality_score
            ), (), disagreement)
        if similarity >= 0.90:
            return ResolutionDecision(
                whisper,
                "equivalent_transcripts_primary_preserved",
                max(whisper_quality.quality_score, gemini_quality.quality_score),
                whisper_quality.reasons,
                disagreement,
            )
        if whisper_quality.is_reliable and not gemini_quality.is_reliable:
            return ResolutionDecision(
                whisper,
                "gemini_failed_whisper_preserved",
                whisper_quality.quality_score,
                whisper_quality.reasons,
                disagreement,
            )
        if gemini_quality.is_reliable and not whisper_quality.is_reliable:
            if _has_fatal_text_problem(whisper_quality) or similarity >= 0.55:
                return ResolutionDecision(
                    gemini,
                    "whisper_low_quality_gemini_selected",
                    gemini_quality.quality_score,
                    gemini_quality.reasons,
                    disagreement,
                )
            return ResolutionDecision(
                whisper,
                "provider_disagreement_primary_preserved",
                whisper_quality.quality_score,
                whisper_quality.reasons,
                disagreement,
            )
        return ResolutionDecision(
            whisper,
            "provider_disagreement_primary_preserved",
            whisper_quality.quality_score,
            whisper_quality.reasons,
            disagreement,
        )


def _has_fatal_text_problem(quality: TranscriptionQuality) -> bool:
    return bool(
        {"empty", "corrupted_characters", "empty_segments", "low_valid_character_ratio"}
        & set(quality.reasons)
    )


def _is_valid_character(char: str) -> bool:
    category = unicodedata.category(char)
    return (
        char.isalnum()
        or category.startswith("L")
        or category.startswith("M")
        or char in "'’-_.,!?/:"
    )


def _has_abnormal_repetition(tokens: list[str]) -> bool:
    consecutive = 1
    for previous, current in zip(tokens, tokens[1:]):
        consecutive = consecutive + 1 if current == previous else 1
        if consecutive >= 4:
            return True
    return (
        len(tokens) >= 6
        and max(tokens.count(token) for token in set(tokens)) / len(tokens) > 0.70
    )


def _has_repeated_character(token: str) -> bool:
    return bool(re.search(r"(.)\1{7,}", token, flags=re.UNICODE))


def _similarity(left: str, right: str) -> float:
    normalize = lambda value: " ".join(value.casefold().split())
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None
