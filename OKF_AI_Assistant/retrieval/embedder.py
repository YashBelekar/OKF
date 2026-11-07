import hashlib
import re

import numpy as np

from config.settings import EMBEDDING_MODEL


def _normalise(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SentenceTransformerEmbedder:
    """Local Sentence Transformers model (downloaded once, then fully offline)."""

    def __init__(self, model_name=EMBEDDING_MODEL):
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Prefer the local cache so a cached model never touches the network.
            # Only if it is not cached yet do we allow a one-time download.
            try:
                self._model = SentenceTransformer(self.name, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.name)
        return self._model

    def encode(self, texts):
        vectors = self._load().encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False,
            batch_size=32,
        )
        return _normalise(vectors.astype(np.float32))


class HashingEmbedder:
    """Dependency-free bag-of-words embedder using signed feature hashing.

    Used as an automatic fallback when sentence-transformers is missing and
    as a deterministic embedder in tests. It matches on shared vocabulary
    rather than meaning, so retrieval is weaker than a real model.
    """

    def __init__(self, dim=384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts):
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
        return _normalise(matrix)


def get_embedder(model_name=EMBEDDING_MODEL):
    """Return the best available embedder and whether it is a fallback."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return HashingEmbedder(), True
    return SentenceTransformerEmbedder(model_name), False
