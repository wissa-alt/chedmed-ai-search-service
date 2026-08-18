
"""Thin, injectable Groq adapter for grounded catalogue answers and transcription."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from groq import Groq

from config import Settings
from models.product import Product

LOGGER = logging.getLogger(__name__)


class GroqClientError(RuntimeError):
    """Raised when the Groq provider cannot return a valid response."""


@dataclass(frozen=True, slots=True)
class GroqTranscription:
    """Real metadata returned by Groq's verbose Whisper response."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class GroqCompletionPort(Protocol):
    """Minimal protocol for the Groq SDK used by this adapter."""

    @property
    def chat(self) -> Any:
        """Expose the SDK chat namespace."""

    @property
    def audio(self) -> Any:
        """Expose the SDK audio namespace."""


class GroqClient:
    """Execute Groq operations without exposing provider details to services."""

    def __init__(
        self,
        settings: Settings,
        client: GroqCompletionPort | None = None,
    ) -> None:
        """Initialise the injected SDK client or create the production client."""
        self._chat_model = settings.groq_chat_model
        self._whisper_model = settings.groq_whisper_model

        try:
            self._client = client or Groq(api_key=settings.groq_api_key)
        except Exception as exc:
            LOGGER.exception("Impossible d'initialiser le client Groq.")
            raise GroqClientError(
                "Le client Groq est indisponible."
            ) from exc

    def understand_query(
        self,
        *,
        model: str,
        temperature: float,
        response_format: dict[str, str],
        messages: list[dict[str, str]],
    ) -> Any:
        """Execute a structured chat completion for query understanding.

        This method keeps the Groq SDK details inside the infrastructure
        adapter. QueryUnderstandingService does not access the SDK directly.
        """
        try:
            return self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format=response_format,
                messages=messages,
            )
        except Exception as exc:
            LOGGER.exception(
                "La compréhension de requête via Groq a échoué."
            )
            raise GroqClientError(
                "Groq n'a pas pu analyser la requête."
            ) from exc

    def generate_catalogue_answer(
        self,
        query: str,
        products: Sequence[Product],
    ) -> str:
        """Generate a French answer grounded only in supplied product snapshots."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        prompt = _catalogue_prompt(query.strip(), products)

        try:
            completion = self._client.chat.completions.create(
                model=self._chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu aides à trouver des produits ChedMed. "
                            "Utilise UNIQUEMENT les résultats catalogue fournis. "
                            "S'il existe au moins un résultat, reconnais obligatoirement "
                            "son existence et ne dis jamais qu'aucun produit n'existe. "
                            "N'invente aucun produit hors de cette liste. Si un résultat "
                            "est seulement approximatif, explique-le explicitement. "
                            "Dis qu'aucun produit pertinent n'a été trouvé uniquement "
                            "lorsque la liste fournie est vide. Réponds en français."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
            )

            content = completion.choices[0].message.content

        except (
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
        ) as exc:
            LOGGER.exception("Réponse Groq de structure invalide.")
            raise GroqClientError(
                "Groq a retourné une réponse invalide."
            ) from exc

        except Exception as exc:
            LOGGER.exception("La génération Groq a échoué.")
            raise GroqClientError(
                "Groq n'a pas pu générer de réponse."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise GroqClientError(
                "Groq a retourné une réponse vide."
            )

        answer = content.strip()
        if products and (
            _claims_no_product(answer)
            or not _mentions_supplied_product(answer, products)
        ):
            LOGGER.warning(
                "Réponse LLM contradictoire avec %d résultat(s); réponse déterministe utilisée.",
                len(products),
            )
            return _deterministic_catalogue_answer(products)
        return answer

    def suggest_seller_description(
        self,
        fields: Mapping[str, str | None],
    ) -> str:
        """Rewrite seller facts without importing attributes from comparables."""
        supplied = {key: value for key, value in fields.items() if value}
        prompt = (
            "Rédige une description marketplace courte (une ou deux phrases) "
            "en utilisant UNIQUEMENT les informations JSON fournies. "
            "N'invente aucune marque, modèle, état, couleur, taille, matière, RAM, "
            "stockage, garantie, ville, accessoire, prix ou caractéristique. "
            "Ne déduis aucun usage. Si les informations sont pauvres, écris simplement "
            "que le produit est proposé à la vente. Retourne uniquement la description.\n"
            f"Données vendeur: {json.dumps(supplied, ensure_ascii=False)}"
        )
        try:
            completion = self._client.chat.completions.create(
                model=self._chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu reformules des annonces vendeur de façon strictement "
                            "factuelle. Toute caractéristique absente est interdite."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            content = completion.choices[0].message.content
        except Exception as exc:
            raise GroqClientError(
                "Groq n'a pas pu suggérer une description vendeur."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise GroqClientError("Groq a retourné une description vendeur vide.")
        return content.strip()

    def transcribe(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
    ) -> GroqTranscription:
        """Transcribe in-memory audio without writing an upload to disk."""
        if not isinstance(audio, bytes) or not audio:
            raise ValueError("Le fichier audio ne peut pas être vide.")

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Le nom du fichier audio est obligatoire.")

        if not isinstance(content_type, str) or not content_type.strip():
            raise ValueError("Le type MIME audio est obligatoire.")

        stream = io.BytesIO(audio)
        stream.name = filename.strip()

        try:
            response = self._client.audio.transcriptions.create(
                file=(
                    stream.name,
                    stream,
                    content_type.strip(),
                ),
                model=self._whisper_model,
                response_format="verbose_json",
                temperature=0.0,
            )

            payload = _provider_payload(response)
            text = payload.get("text")

        except (
            AttributeError,
            KeyError,
            TypeError,
        ) as exc:
            LOGGER.exception(
                "Réponse de transcription Groq invalide."
            )
            raise GroqClientError(
                "Groq a retourné une transcription invalide."
            ) from exc

        except Exception as exc:
            LOGGER.exception(
                "La transcription Groq a échoué."
            )
            raise GroqClientError(
                "Groq n'a pas pu transcrire l'audio."
            ) from exc

        if not isinstance(text, str) or not text.strip():
            raise GroqClientError(
                "Groq a retourné une transcription vide."
            )

        language = payload.get("language")
        duration = _finite_float(payload.get("duration"))
        segments = tuple(
            segment
            for raw_segment in payload.get("segments", ()) or ()
            if (segment := _provider_payload(raw_segment))
        )
        metadata = {
            key: payload[key]
            for key in ("task", "model")
            if key in payload and payload[key] is not None
        }
        return GroqTranscription(
            text=text.strip(),
            language=language.strip() if isinstance(language, str) and language.strip() else None,
            duration=duration,
            segments=segments,
            metadata=metadata,
        )


def _catalogue_prompt(
    query: str,
    products: Sequence[Product],
) -> str:
    """Build a compact, structured prompt from typed catalogue data."""

    if not products:
        catalogue = "Aucun produit correspondant n'est disponible."

    else:
        catalogue = "\n".join(
            (
                f"- ID: {product.id}; "
                f"titre: {product.title}; "
                f"description: {product.description}; "
                f"catégorie: {product.category}; "
                f"prix: {product.price} {product.currency}; "
                f"ville: {product.city or 'non renseignée'}; "
                f"état: {product.condition or 'non renseigné'}"
            )
            for product in products
        )

    count = len(products)
    return (
        f"Demande utilisateur : {query}\n"
        f"Nombre exact de résultats fournis : {count}\n\n"
        f"Catalogue disponible :\n{catalogue}"
    )


def _provider_payload(value: Any) -> dict[str, Any]:
    """Convert SDK models or mappings without inventing absent metadata."""
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, dict) else {}
    fields = ("text", "language", "duration", "segments", "task", "model")
    return {
        field_name: field_value
        for field_name in fields
        if (field_value := getattr(value, field_name, None)) is not None
    }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _claims_no_product(answer: str) -> bool:
    normalized = " ".join(answer.casefold().split())
    markers = (
        "aucun produit",
        "pas de produit",
        "il n'y a pas de produit",
        "il n’existe pas de produit",
        "no product",
    )
    return any(marker in normalized for marker in markers)


def _deterministic_catalogue_answer(products: Sequence[Product]) -> str:
    details = "; ".join(
        f"{product.title} (ID {product.id}, {product.price} {product.currency})"
        for product in products
    )
    return f"J’ai trouvé {len(products)} résultat(s) dans le catalogue : {details}."


def _mentions_supplied_product(answer: str, products: Sequence[Product]) -> bool:
    normalized = answer.casefold()
    return any(
        product.title.casefold() in normalized
        or str(product.id).casefold() in normalized
        for product in products
    )
