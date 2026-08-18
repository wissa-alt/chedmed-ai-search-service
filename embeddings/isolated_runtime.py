"""Lifecycle-safe subprocess runtime used to isolate PyTorch from FAISS."""

from __future__ import annotations

import pickle
import subprocess
import sys

import numpy as np

from embeddings.embedder import EmbeddingServiceError


class IsolatedEmbeddingRuntime:
    """Keep one SentenceTransformer in a FAISS-free child process."""

    def __init__(self, model_name: str, device: str) -> None:
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "embeddings.embedding_worker", model_name, device],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert self._process.stdin is not None and self._process.stdout is not None
            self._input = self._process.stdin
            self._output = self._process.stdout
            startup = pickle.load(self._output)
            if startup != {"ready": True}:
                assert self._process.stderr is not None
                diagnostics = self._process.stderr.read().decode("utf-8", errors="replace")
                raise EmbeddingServiceError(f"Le worker d'embeddings n'a pas démarré : {startup}; {diagnostics}")
        except Exception as exc:
            self._terminate()
            raise EmbeddingServiceError("Impossible de démarrer le worker d'embeddings.") from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts in the isolated PyTorch process."""
        try:
            pickle.dump({"operation": "encode", "texts": texts}, self._input)
            self._input.flush()
            print("Parent : attente de la réponse du worker...", flush=True)
            result = pickle.load(self._output)
            print("Parent : réponse reçue.", flush=True)
        except (EOFError, OSError) as exc:
            raise EmbeddingServiceError("Le worker d'embeddings s'est arrêté inopinément.") from exc
        if isinstance(result, dict) and "error" in result:
            raise EmbeddingServiceError(f"Le worker d'embeddings a échoué : {result['error']}")
        return np.asarray(result, dtype=np.float32)

    def close(self) -> None:
        """Stop the child cleanly without multiprocessing queues or semaphores."""
        try:
            pickle.dump({"operation": "close"}, self._input)
            self._input.flush()
        except (EOFError, OSError, BrokenPipeError):
            pass
        self._input.close()
        self._output.close()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)

    def _terminate(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
