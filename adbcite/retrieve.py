"""Three ways to find passages: keyword (BM25), dense (embeddings) and hybrid."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .embed import tokenize
from .ingest import Chunk, load_chunks, save_chunks

MODES = ("bm25", "dense", "hybrid")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Index:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder):
        self.chunks = chunks
        self.embeddings = embeddings          # one unit-length row per chunk
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    # ---- building and saving -------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk], embedder, batch_size: int = 8) -> "Index":
        parts = [embedder.encode([c.text for c in chunks[i:i + batch_size]])
                 for i in range(0, len(chunks), batch_size)]
        return cls(chunks, np.vstack(parts), embedder)

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        save_chunks(self.chunks, index_dir / "chunks.jsonl")
        np.save(index_dir / "embeddings.npy", self.embeddings)
        (index_dir / "meta.json").write_text(json.dumps({"embedder": self.embedder.name}))

    @classmethod
    def load(cls, index_dir: Path, embedder) -> "Index":
        index_dir = Path(index_dir)
        return cls(load_chunks(index_dir / "chunks.jsonl"),
                   np.load(index_dir / "embeddings.npy"), embedder)

    # ---- searching -------------------------------------------------------------
    def _hits(self, order: np.ndarray, scores: np.ndarray, k: int) -> list[Hit]:
        return [Hit(self.chunks[i], float(scores[i]), rank)
                for rank, i in enumerate(order[:k], start=1)]

    def search_bm25(self, query: str, k: int = 5) -> list[Hit]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        return self._hits(np.argsort(-scores, kind="stable"), scores, k)

    def dense_scores(self, query: str) -> np.ndarray:
        return self.embeddings @ self.embedder.encode([query])[0]   # cosine similarity

    def search_dense(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.dense_scores(query)
        return self._hits(np.argsort(-scores, kind="stable"), scores, k)

    def search_hybrid(self, query: str, k: int = 5, pool: int = 50, rrf_k: int = 60) -> list[Hit]:
        """Reciprocal rank fusion of the BM25 and dense rankings."""
        fused: dict[int, float] = {}
        for scores in (np.asarray(self.bm25.get_scores(tokenize(query))), self.dense_scores(query)):
            for rank, i in enumerate(np.argsort(-scores, kind="stable")[:pool], start=1):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank)
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], rank) for rank, i in enumerate(order, start=1)]

    def search(self, query: str, mode: str = "hybrid", k: int = 5) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        return getattr(self, f"search_{mode}")(query, k)
