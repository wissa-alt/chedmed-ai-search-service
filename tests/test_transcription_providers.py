"""Unit tests for in-memory Whisper and Gemini provider adapters."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from config import Settings
from llm.groq_client import GroqTranscription
from llm.transcription_providers import GeminiAudioTranscriptionProvider, WhisperTranscriptionProvider
from services.audio_errors import AudioProviderError


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test", host="127.0.0.1", port=5000, log_level="CRITICAL",
        chedmed_webhook_secret="secret", groq_api_key="groq", gemini_api_key="gemini",
        project_root=tmp_path,
    )


def test_whisper_provider_returns_structured_candidate() -> None:
    client = Mock()
    client.transcribe.return_value = "bghit sberdila"
    result = WhisperTranscriptionProvider(client).transcribe(
        b"audio", "voice.wav", "audio/wav"
    )
    assert result.provider == "whisper"
    assert result.text == "bghit sberdila"


def test_whisper_provider_preserves_real_verbose_metadata() -> None:
    client = Mock()
    client.transcribe.return_value = GroqTranscription(
        "bghit laptop",
        language="ar",
        duration=2.5,
        segments=(
            {"text": "bghit laptop", "avg_logprob": -0.4, "no_speech_prob": 0.05},
        ),
    )
    result = WhisperTranscriptionProvider(client).transcribe(
        b"audio", "voice.wav", "audio/wav"
    )
    assert result.language_hint == "ar"
    assert result.metadata["duration"] == 2.5
    assert result.metadata["avg_logprob"] == -0.4
    assert result.metadata["max_no_speech_prob"] == 0.05


def test_gemini_receives_original_bytes_and_strict_prompt(settings: Settings) -> None:
    client = Mock()
    client.models.generate_content.return_value = Mock(text="bghit parfum dial l3yalat")
    result = GeminiAudioTranscriptionProvider(settings, client).transcribe(
        b"original-audio",
        "voice.mp3",
        "audio/mpeg",
        whisper_text="bghit barfan dial l3yalat",
    )
    call = client.models.generate_content.call_args.kwargs
    assert result.provider == "gemini"
    assert call["model"] == settings.gemini_audio_model
    assert "Do NOT rewrite" in call["contents"][0]
    assert "Do NOT translate" in call["contents"][0]
    assert "unknown brand" in call["contents"][0]
    assert "bghit barfan dial l3yalat" in call["contents"][0]
    assert call["contents"][1].inline_data.data == b"original-audio"
    assert call["contents"][1].inline_data.mime_type == "audio/mp3"


def test_gemini_empty_or_unsupported_response_is_controlled(settings: Settings) -> None:
    client = Mock()
    client.models.generate_content.return_value = Mock(text=" ")
    provider = GeminiAudioTranscriptionProvider(settings, client)
    with pytest.raises(AudioProviderError, match="vide"):
        provider.transcribe(b"audio", "voice.wav", "audio/wav")
    with pytest.raises(AudioProviderError, match="prend pas en charge"):
        provider.transcribe(b"audio", "voice.m4a", "audio/x-m4a")


def test_gemini_search_normalization_uses_audio_without_inventing_constraints(
    settings: Settings,
) -> None:
    """Voice recovery has an explicit semantic-search prompt, not a product picker."""
    client = Mock()
    client.models.generate_content.return_value = Mock(text="baskets pour hommes")
    result = GeminiAudioTranscriptionProvider(settings, client).normalize_search(
        b"same-original-audio",
        "voice.wav",
        "audio/wav",
        whisper_text="بغيت سبرتي لا ديال الرجال",
    )

    prompt, audio_part = client.models.generate_content.call_args.kwargs["contents"]
    assert result.text == "baskets pour hommes"
    assert "marketplace search query" in prompt
    assert "Do NOT choose or recommend a product" in prompt
    assert "Do NOT invent a brand" in prompt
    assert "بغيت سبرتي لا ديال الرجال" in prompt
    assert audio_part.inline_data.data == b"same-original-audio"


def test_gemini_provider_preserves_darija_instead_of_translating(
    settings: Settings,
) -> None:
    client = Mock()
    client.models.generate_content.return_value = Mock(
        text="bghit laptop noir f casa"
    )
    result = GeminiAudioTranscriptionProvider(settings, client).transcribe(
        b"same-original-audio", "voice.wav", "audio/wav"
    )
    assert result.text == "bghit laptop noir f casa"
    assert "Je cherche" not in result.text
    part = client.models.generate_content.call_args.kwargs["contents"][1]
    assert part.inline_data.data == b"same-original-audio"
