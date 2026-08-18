"""Unit tests for provider strategy, validation, quality, and resolution."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from models.transcription import TranscriptionCandidate
from services.audio_errors import AudioFallbackError, AudioProviderError, InvalidAudioError
from services.audio_transcription_service import AudioTranscriptionService, AudioValidationPolicy
from services.transcription_quality import TranscriptionQualityEvaluator, TranscriptionResolver


@pytest.fixture
def policy() -> AudioValidationPolicy:
    return AudioValidationPolicy(20, frozenset({"audio/wav", "audio/mpeg"}))


def provider(name: str, text: str, metadata: dict[str, object] | None = None) -> Mock:
    mock = Mock(name=name)
    mock.name = name
    mock.transcribe.return_value = TranscriptionCandidate(
        name, text, metadata=metadata or {}
    )
    return mock


def service(whisper: Mock | None, gemini: Mock | None, policy: AudioValidationPolicy, mode: str) -> AudioTranscriptionService:
    return AudioTranscriptionService(
        whisper, gemini, mode=mode, validation_policy=policy
    )


def test_fallback_is_not_called_for_good_whisper(policy: AudioValidationPolicy) -> None:
    whisper = provider("whisper", "bghit sberdila dial rjal")
    gemini = provider("gemini", "unused")
    result = service(whisper, gemini, policy, "fallback").transcribe(
        b"audio", "voice.wav", "audio/wav"
    )
    assert result.text == "bghit sberdila dial rjal"
    assert result.primary_provider == "whisper"
    assert result.used_fallback is False
    gemini.transcribe.assert_not_called()


def test_whisper_failure_triggers_gemini(policy: AudioValidationPolicy) -> None:
    whisper = provider("whisper", "unused")
    whisper.transcribe.side_effect = AudioProviderError("down")
    gemini = provider("gemini", "bghit parfum dial l3yalat")
    result = service(whisper, gemini, policy, "fallback").transcribe(
        b"audio", "voice.wav", "audio/wav"
    )
    assert result.text == "bghit parfum dial l3yalat"
    assert result.used_fallback is True
    assert result.resolution_reason == "whisper_failed_gemini_selected"


def test_low_quality_whisper_triggers_gemini(policy: AudioValidationPolicy) -> None:
    whisper = provider("whisper", "����")
    gemini = provider("gemini", "9leb lia 3la laptop gaming")
    result = service(whisper, gemini, policy, "fallback").transcribe(
        b"audio", "voice.wav", "audio/wav"
    )
    assert result.primary_provider == "gemini"
    assert result.whisper_text == "����"
    assert result.quality_score == 1.0


def test_real_whisper_metadata_can_trigger_conditional_fallback(
    policy: AudioValidationPolicy,
) -> None:
    whisper = provider(
        "whisper",
        "bghit labtop geming ta7t 6000",
        {"avg_logprob": -2.0, "max_no_speech_prob": 0.8},
    )
    gemini = provider("gemini", "bghit laptop gaming ta7t 6000 dh")
    result = service(whisper, gemini, policy, "fallback").transcribe(
        b"audio", "voice.wav", "audio/wav"
    )
    assert result.text == "bghit laptop gaming ta7t 6000 dh"
    assert result.primary_provider == "gemini"
    assert result.resolution_reason == "whisper_low_quality_gemini_selected"
    gemini.transcribe.assert_called_once_with(
        b"audio",
        "voice.wav",
        "audio/wav",
        whisper_text="bghit labtop geming ta7t 6000",
    )


def test_gemini_failure_preserves_reliable_whisper_in_dual(
    policy: AudioValidationPolicy,
) -> None:
    whisper = provider("whisper", "bghit iphone 15 noir f casa")
    gemini = provider("gemini", "unused")
    gemini.transcribe.side_effect = AudioProviderError("down")
    result = service(whisper, gemini, policy, "dual").transcribe(
        b"audio", "voice.wav", "audio/wav"
    )
    assert result.text == "bghit iphone 15 noir f casa"
    assert result.resolution_reason == "gemini_failed_whisper_preserved"


def test_both_low_quality_candidates_raise_typed_error(
    policy: AudioValidationPolicy,
) -> None:
    whisper = provider("whisper", "����")
    gemini = provider("gemini", "....")
    with pytest.raises(AudioFallbackError):
        service(whisper, gemini, policy, "dual").transcribe(
            b"audio", "voice.wav", "audio/wav"
        )


def test_both_provider_failures_are_controlled(policy: AudioValidationPolicy) -> None:
    whisper = provider("whisper", "unused")
    gemini = provider("gemini", "unused")
    whisper.transcribe.side_effect = AudioProviderError("down")
    gemini.transcribe.side_effect = AudioProviderError("down")
    with pytest.raises(AudioFallbackError):
        service(whisper, gemini, policy, "fallback").transcribe(
            b"audio", "voice.wav", "audio/wav"
        )


def test_gemini_only_success_and_failure(policy: AudioValidationPolicy) -> None:
    gemini = provider("gemini", "قلب ليا على تليفون سامسونغ")
    audio_service = service(None, gemini, policy, "gemini")
    assert audio_service.transcribe(b"audio", "voice.wav", "audio/wav").text.startswith("قلب")
    gemini.transcribe.side_effect = AudioProviderError("down")
    with pytest.raises(AudioProviderError):
        audio_service.transcribe(b"audio", "voice.wav", "audio/wav")


def test_alternative_transcription_calls_only_gemini_with_original_audio(
    policy: AudioValidationPolicy,
) -> None:
    whisper = provider("whisper", "primary words")
    gemini = provider("gemini", "independent words")
    result = service(whisper, gemini, policy, "fallback").transcribe_alternative(
        b"same-original-audio", "voice.wav", "audio/wav", provider="gemini"
    )
    assert result.text == "independent words"
    assert result.primary_provider == "gemini"
    gemini.transcribe.assert_called_once_with(
        b"same-original-audio", "voice.wav", "audio/wav", whisper_text=None
    )
    whisper.transcribe.assert_not_called()


def test_dual_calls_both_and_preserves_equivalent_primary(policy: AudioValidationPolicy) -> None:
    whisper = provider("whisper", "je cherche laptop gaming à casa")
    gemini = provider("gemini", "je cherche laptop gaming à Casa")
    result = service(whisper, gemini, policy, "dual").transcribe(
        b"audio", "voice.wav", "audio/wav"
    )
    assert result.text == "je cherche laptop gaming à casa"
    assert result.resolution_reason == "equivalent_transcripts_primary_preserved"
    assert result.used_fallback is False
    whisper.transcribe.assert_called_once()
    gemini.transcribe.assert_called_once()


@pytest.mark.parametrize(
    "audio,filename,mime",
    [
        (b"", "voice.wav", "audio/wav"),
        (b"audio", "", "audio/wav"),
        (b"audio", "voice.wav", "text/plain"),
        (b"x" * 21, "voice.wav", "audio/wav"),
    ],
)
def test_invalid_audio_is_rejected_before_provider(
    policy: AudioValidationPolicy, audio: bytes, filename: str, mime: str
) -> None:
    whisper = provider("whisper", "unused")
    with pytest.raises(InvalidAudioError):
        service(whisper, None, policy, "whisper").transcribe(audio, filename, mime)
    whisper.transcribe.assert_not_called()


@pytest.mark.parametrize(
    "text",
    [
        "bghit sberdila dial rjal",
        "bghit parfum dial l3yalat",
        "9leb lia 3la laptop gaming",
        "bghit telephone samsung k7el",
        "ma yfoutch 3000 dirham",
        "بغيت سبرديلة ديال الرجال",
        "قلب ليا على تليفون سامسونغ",
        "je cherche laptop gaming à Casablanca",
        "I need black sneakers for men",
        "bghit iphone 15 pro f casa",
        "bghit pc ta7t 5000 dh",
        "je cherche chi téléphone mzyan",
        "I need pc gaming f casa",
        "kan9leb 3la laptop sghir pour montage budget 6000 dh",
        "bghit Zyrqophone X99 f Ouarzazate",
        "find AcmeNova model QX-71 under 4200 MAD",
    ],
)
def test_quality_evaluator_accepts_multilingual_darija(text: str) -> None:
    assessment = TranscriptionQualityEvaluator().evaluate(
        TranscriptionCandidate("whisper", text)
    )
    assert assessment.acceptable is True


def test_resolver_never_fuses_or_translates_disagreement() -> None:
    whisper = TranscriptionCandidate("whisper", "bghit parfum dial l3yalat")
    gemini = TranscriptionCandidate("gemini", "je cherche un parfum pour femmes")
    decision = TranscriptionResolver().resolve(whisper, gemini)
    assert decision.candidate is not None
    assert decision.candidate.text == whisper.text
    assert decision.reason == "provider_disagreement_primary_preserved"
    assert decision.disagreement == {"similarity": 0.351, "strong": True}


def test_translated_gemini_hypothesis_does_not_replace_darija_primary() -> None:
    whisper = TranscriptionCandidate(
        "whisper",
        "bghit laptop noir f casa",
        metadata={"avg_logprob": -1.4},
    )
    translated = TranscriptionCandidate(
        "gemini", "Je cherche un ordinateur portable noir à Casablanca"
    )
    decision = TranscriptionResolver().resolve(whisper, translated)
    assert decision.candidate is whisper
    assert decision.reason == "provider_disagreement_primary_preserved"


def test_resolver_never_forces_unknown_vocabulary() -> None:
    primary = TranscriptionCandidate("whisper", "bghit Zyrqophone X99")
    known_word_guess = TranscriptionCandidate("gemini", "bghit smartphone Samsung")
    decision = TranscriptionResolver().resolve(primary, known_word_guess)
    assert decision.candidate is primary
    assert "Zyrqophone" in decision.candidate.text
