"""Dense retrieval: sentence embeddings + a NumPy cosine index with disk cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from app.search.dinos import Dino

Aggregation = Literal["max", "top2_mean"]


class Embedder:
    """Wraps a SentenceTransformer, adding E5 `query:`/`passage:` prefixes when needed."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._e5 = "e5" in model_name.lower()
        self._model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if self._e5:
            texts = [f"{prefix}: {t}" for t in texts]
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode user queries to unit vectors."""
        return self._encode(texts, "query")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Encode indexed passages to unit vectors."""
        return self._encode(texts, "passage")


class DenseIndex:
    """Cosine index over every passage of every dino; scores aggregate per dino.

    ponytail: brute-force NumPy matmul, fine for ~150 vectors. Swap `_vectors @ q`
    for a FAISS/pgvector lookup if the base grows to many thousands.
    """

    def __init__(self, embedder: Embedder, dinos: list[Dino], cache_dir: Path | None = None) -> None:
        self.embedder = embedder
        self.n_dinos = len(dinos)
        texts = [p for d in dinos for p in d.passages]
        self._owner = np.array([i for i, d in enumerate(dinos) for _ in d.passages])
        self._vectors = self._load_or_encode(texts, cache_dir)

    def _load_or_encode(self, texts: list[str], cache_dir: Path | None) -> np.ndarray:
        if cache_dir is None:
            return self.embedder.encode_passages(texts)
        key = hashlib.sha256("\n".join([self.embedder.model_name, *texts]).encode()).hexdigest()[:16]
        path = cache_dir / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.embedder.encode_passages(texts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def scores(self, query_vec: np.ndarray, aggregation: Aggregation = "top2_mean") -> np.ndarray:
        """Return one cosine score per dino (shape: n_dinos)."""
        sims = self._vectors @ query_vec
        out = np.empty(self.n_dinos, dtype=np.float32)
        for i in range(self.n_dinos):
            s = np.sort(sims[self._owner == i])[::-1]
            out[i] = s[0] if aggregation == "max" else s[:2].mean()
        return out
