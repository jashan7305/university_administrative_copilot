"""Semantic search over the procedure knowledge base.

Three retrievers share one interface, ``search(query, k, intent=None) -> list[Hit]``:
  * BM25Retriever   - lexical (Okapi BM25) over preprocessed tokens
  * DenseRetriever  - sentence-embedding cosine similarity (bi-encoder)
  * HybridRetriever - Reciprocal Rank Fusion of both (Cormack et al., 2009)

Results are aggregated from chunks to documents (max score per document), because the service
request is built from a whole procedure. An optional predicted intent softly boosts documents of
that intent, so a classifier mistake cannot completely hide the right document.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from logic.intent.preprocess import PreprocessConfig, Preprocessor
from logic.knowledge.store import Chunk, load_chunks

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
_bm25_prep = Preprocessor(PreprocessConfig(spell_correct=False, remove_stopwords=True, lemmatize=True))


@dataclass
class Hit:
    doc_id: str
    score: float
    chunk: Chunk


@lru_cache(maxsize=4)
def get_encoder(name: str = DEFAULT_EMBEDDER):
    from sentence_transformers import SentenceTransformer  # heavy import, deferred

    return SentenceTransformer(name, device="cpu")


def _to_docs(chunks, scores, k: int, intent: str | None, boost: float) -> list[Hit]:
    best: dict[str, Hit] = {}
    for c, s in zip(chunks, scores):
        s = float(s) * (boost if intent and c.intent == intent else 1.0)
        if c.doc_id not in best or s > best[c.doc_id].score:
            best[c.doc_id] = Hit(c.doc_id, s, c)
    return sorted(best.values(), key=lambda h: -h.score)[:k]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: tuple[Chunk, ...] | None = None, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks or load_chunks()
        self.bm25 = BM25Okapi([_bm25_prep.tokens(c.indexed_text) for c in self.chunks], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return np.maximum(self.bm25.get_scores(_bm25_prep.tokens(query)), 0.0)  # BM25Okapi idf can go negative

    def search(self, query: str, k: int = 5, intent: str | None = None, boost: float = 1.3) -> list[Hit]:
        return _to_docs(self.chunks, self.scores(query), k, intent, boost)


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: tuple[Chunk, ...] | None = None, model: str = DEFAULT_EMBEDDER):
        self.chunks = chunks or load_chunks()
        self.model_name = model
        self.encoder = get_encoder(model)
        self.matrix = self.encoder.encode([c.indexed_text for c in self.chunks], normalize_embeddings=True,
                                          batch_size=64, show_progress_bar=False)

    def scores(self, query: str) -> np.ndarray:
        q = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.matrix @ q

    def search(self, query: str, k: int = 5, intent: str | None = None, boost: float = 1.15) -> list[Hit]:
        return _to_docs(self.chunks, self.scores(query), k, intent, boost)


class HybridRetriever:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (rrf_k + rank_r(d)), computed at document level."""

    name = "hybrid"

    def __init__(self, bm25: BM25Retriever | None = None, dense: DenseRetriever | None = None, rrf_k: int = 60):
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5, intent: str | None = None, boost: float = 1.2) -> list[Hit]:
        fused: dict[str, float] = {}
        best_chunk: dict[str, Chunk] = {}
        for retriever in (self.dense, self.bm25):  # dense first, so its passage is kept for generation
            for rank, hit in enumerate(retriever.search(query, k=len(self.bm25.chunks), intent=None), start=1):
                fused[hit.doc_id] = fused.get(hit.doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
                best_chunk.setdefault(hit.doc_id, hit.chunk)
        hits = [Hit(d, s * (boost if intent and best_chunk[d].intent == intent else 1.0), best_chunk[d]) for d, s in fused.items()]
        return sorted(hits, key=lambda h: -h.score)[:k]
