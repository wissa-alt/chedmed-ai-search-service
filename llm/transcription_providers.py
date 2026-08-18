"""Provider adapters that turn original in-memory audio into typed candidates."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from google import genai
from google.genai import types

from config import Settings
from llm.groq_client import GroqTranscription
from models.transcription import TranscriptionCandidate
from services.audio_errors import AudioProviderError

LOGGER = logging.getLogger(__name__)

_GEMINI_TRANSCRIPTION_PROMPT = """You are reviewing an automatic speech transcription.

Listen carefully to the ORIGINAL AUDIO.

Whisper transcription:
{whisper_text}

Your task is to return the most faithful literal transcription of what the speaker actually said.

Do NOT translate.
Do NOT summarize.
Do NOT answer the user.
Do NOT infer user intent.
Do NOT rewrite.
Do NOT normalize into French or English.
Do NOT improve grammar.
Do NOT invent missing words.
Do NOT choose products.
Do NOT replace an unknown brand, product, or term with a catalogue term just because it looks similar.

Preserve Moroccan Darija, Arabic, Arabizi, French, English, code-switching,
brands, model names, numbers, prices, cities, and unknown words if uncertain.
Correct Whisper only when the original audio clearly supports the correction.

Return ONLY the corrected literal transcription."""

_GEMINI_SEARCH_NORMALIZATION_PROMPT = """You are preparing a short marketplace search query from speech.

Listen carefully to the ORIGINAL AUDIO and use the Whisper text only as a fallible hint.

Whisper transcription:
{whisper_text}

Return one concise search query that preserves what the speaker requested.
You MAY normalize Moroccan Darija, Arabic, Arabizi, French, English, and mixed speech
into clear product-search wording when the audio supports it.

Do NOT answer the user.
Do NOT choose or recommend a product.
Do NOT use or guess catalogue contents.
Do NOT invent a brand, colour, price, city, audience, condition, model, or other
constraint that was not spoken.
Preserve every explicit brand, model, number, price, city, colour, and audience.
Keep unknown terms when the audio does not clearly support a normalization.

Return ONLY the normalized search query."""

_GEMINI_MIME_ALIASES = {
    "audio/mpeg": "audio/mp3",
    "audio/x-wav": "audio/wav",
    "audio/x-flac": "audio/flac",
}


class RawTranscriptionPort(Protocol):
    """Minimal contract implemented by the existing Groq adapter."""

    def transcribe(
        self, audio: bytes, filename: str, content_type: str
    ) -> str | GroqTranscription:
        """Return provider text and any real Whisper metadata available."""


class WhisperTranscriptionProvider:
    """Adapt the existing Groq Whisper client to the typed provider contract."""

    name = "whisper"

    def __init__(self, client: RawTranscriptionPort) -> None:
        self._client = client

    def transcribe(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        *,
        whisper_text: str | None = None,
    ) -> TranscriptionCandidate:
        try:
            response = self._client.transcribe(audio, filename, content_type)
        except Exception as exc:
            raise AudioProviderError("Whisper n'a pas pu transcrire l'audio.") from exc
        if isinstance(response, str):
            return _candidate(self.name, response)
        segment_metadata = tuple(_whisper_segment(segment) for segment in response.segments)
        avg_logprobs = [
            value
            for segment in segment_metadata
            if (value := segment.get("avg_logprob")) is not None
        ]
        no_speech_probs = [
            value
            for segment in segment_metadata
            if (value := segment.get("no_speech_prob")) is not None
        ]
        metadata: dict[str, Any] = {
            **response.metadata,
            "segments": segment_metadata,
            "segment_count": len(segment_metadata),
        }
        if response.duration is not None:
            metadata["duration"] = response.duration
        if avg_logprobs:
            metadata["avg_logprob"] = sum(avg_logprobs) / len(avg_logprobs)
        if no_speech_probs:
            metadata["max_no_speech_prob"] = max(no_speech_probs)
        return TranscriptionCandidate(
            provider=self.name,
            text=response.text.strip(),
            language_hint=response.language,
            metadata=metadata,
        )


class GeminiAudioTranscriptionProvider:
    """Transcribe the original bytes with the official Google Gen AI SDK."""

    name = "gemini"

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
    ) -> None:
        self._model = settings.gemini_audio_model
        self._client = client
        self._api_key = settings.gemini_api_key

    def transcribe(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        *,
        whisper_text: str | None = None,
    ) -> TranscriptionCandidate:
        return self._generate(
            audio,
            filename,
            content_type,
            whisper_text=whisper_text,
            prompt=_GEMINI_TRANSCRIPTION_PROMPT,
        )

    def normalize_search(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        *,
        whisper_text: str | None = None,
    ) -> TranscriptionCandidate:
        """Use the original audio to produce a concise, non-catalogue search text."""
        return self._generate(
            audio,
            filename,
            content_type,
            whisper_text=whisper_text,
            prompt=_GEMINI_SEARCH_NORMALIZATION_PROMPT,
        )

    def _generate(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        *,
        whisper_text: str | None,
        prompt: str,
    ) -> TranscriptionCandidate:
        del filename
        mime_type = _GEMINI_MIME_ALIASES.get(content_type, content_type)
        if mime_type not in {
            "audio/wav", "audio/mp3", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"
        }:
            raise AudioProviderError(
                f"Gemini ne prend pas en charge le type MIME {content_type}."
            )
        if self._client is None:
            if not self._api_key:
                raise AudioProviderError("GEMINI_API_KEY est absente.")
            try:
                self._client = genai.Client(api_key=self._api_key)
            except Exception as exc:
                raise AudioProviderError("Le client Gemini est indisponible.") from exc
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    prompt.format(
                        whisper_text=whisper_text or "[unavailable]"
                    ),
                    types.Part.from_bytes(data=audio, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=2048,
                ),
            )
            text = response.text
        except AudioProviderError:
            raise
        except Exception as exc:
            LOGGER.exception("La transcription Gemini a échoué.")
            raise AudioProviderError("Gemini n'a pas pu transcrire l'audio.") from exc
        return _candidate(self.name, text)


def _candidate(provider: str, text: Any) -> TranscriptionCandidate:
    if not isinstance(text, str) or not text.strip():
        raise AudioProviderError(
            f"{provider.capitalize()} a retourné une transcription vide ou invalide."
        )
    return TranscriptionCandidate(provider=provider, text=text.strip())


def _whisper_segment(segment: dict[str, Any]) -> dict[str, Any]:
    """Keep only real, non-sensitive fields returned for one Whisper segment."""
    fields = (
        "id",
        "start",
        "end",
        "text",
        "avg_logprob",
        "no_speech_prob",
        "compression_ratio",
        "temperature",
    )
    result = {field: segment[field] for field in fields if segment.get(field) is not None}
    for numeric_field in ("avg_logprob", "no_speech_prob"):
        value = result.get(numeric_field)
        if value is not None:
            try:
                result[numeric_field] = float(value)
            except (TypeError, ValueError):
                result.pop(numeric_field)
    return result
