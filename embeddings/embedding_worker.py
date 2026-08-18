"""Dedicated SentenceTransformer process, isolated from FAISS native libraries."""

from __future__ import annotations

import pickle
import sys
import traceback
from typing import Any

import numpy as np


def _log(message: str) -> None:
    """Write diagnostic messages immediately to stderr."""
    print(message, file=sys.stderr, flush=True)


def main() -> int:
    """Serve encoding requests received through standard input/output pipes."""

    if len(sys.argv) != 3:
        _log("Worker : arguments invalides.")
        return 1

    model_name = sys.argv[1]
    device = sys.argv[2]

    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer

    try:
        _log(f"Worker : chargement du modèle '{model_name}' sur '{device}'...")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)

        _log("Worker : modèle chargé.")

        pickle.dump({"ready": True}, output_stream)
        output_stream.flush()

        _log("Worker : prêt à recevoir des requêtes.")

        while True:
            try:
                request: dict[str, Any] = pickle.load(input_stream)
            except EOFError:
                _log("Worker : pipe fermé par le processus parent.")
                return 0

            if not isinstance(request, dict):
                _log("Worker : requête invalide.")
                continue

            operation = request.get("operation")

            if operation == "close":
                _log("Worker : fermeture demandée.")
                return 0

            if operation != "encode":
                _log(f"Worker : opération inconnue : {operation}")
                pickle.dump(
                    {"error": f"Opération inconnue : {operation}"},
                    output_stream,
                )
                output_stream.flush()
                continue

            texts = request.get("texts")

            if not isinstance(texts, list):
                _log("Worker : champ 'texts' invalide.")
                pickle.dump(
                    {"error": "Le champ 'texts' doit être une liste."},
                    output_stream,
                )
                output_stream.flush()
                continue

            _log(f"Worker : requête reçue ({len(texts)} texte(s)).")

            try:
                encoded = model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                _log("Worker : embedding terminé.")

                pickle.dump(
                    np.asarray(encoded, dtype=np.float32),
                    output_stream,
                )
                output_stream.flush()

                _log("Worker : réponse envoyée.")

            except Exception:
                traceback.print_exc(file=sys.stderr)

                pickle.dump(
                    {"error": "Erreur pendant model.encode()."},
                    output_stream,
                )
                output_stream.flush()

    except Exception:
        traceback.print_exc(file=sys.stderr)

        try:
            pickle.dump(
                {"error": "Impossible d'initialiser SentenceTransformer."},
                output_stream,
            )
            output_stream.flush()
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())