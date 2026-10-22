"""Text embedders.

``SentenceTransformerEmbedder`` is the real one. ``HashingEmbedder`` is a tiny,
dependency-free stand-in used by the unit tests so they run without downloading
a model. It is NOT a semantic model and must not be used for reported results.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(name: str):
    return HashingEmbedder() if name == "hashing" else SentenceTransformerEmbedder(name)
