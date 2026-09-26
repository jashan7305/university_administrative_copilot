"""Intent classifiers on sentence embeddings, and a probability-averaging ensemble with the TF-IDF baseline.

Embeddings capture meaning beyond shared words, which targets the Week 2 failure mode: queries whose
wording never appeared in training (e.g. "I failed physics, how do I reappear?").
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from logic.intent.preprocess import PreprocessConfig, Preprocessor
from logic.retrieval.search import get_encoder

# Embedding models see near-raw text: only contractions / abbreviations are expanded (no stemming, no masking).
_light = Preprocessor(PreprocessConfig(lowercase=True, expand_contractions=True, expand_abbreviations=True,
                                       mask_entities=False, spell_correct=False, remove_stopwords=False, lemmatize=False))

EMBEDDERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
}


class EmbeddingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embedder: str = "minilm", C: float = 10.0):
        self.embedder = embedder
        self.C = C

    def _embed(self, texts) -> np.ndarray:
        enc = get_encoder(EMBEDDERS[self.embedder])
        return enc.encode(_light.transform(list(texts)), normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    def fit(self, X, y):
        self.clf_ = LogisticRegression(C=self.C, max_iter=5000, class_weight="balanced").fit(self._embed(X), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.clf_.predict_proba(self._embed(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Weighted average of class probabilities from already-configured members (same label set)."""

    def __init__(self, members: list, weights: list[float] | None = None):
        self.members = members
        self.weights = weights

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        self.classes_ = self.members[0].classes_
        assert all((m.classes_ == self.classes_).all() for m in self.members)
        return self

    def predict_proba(self, X) -> np.ndarray:
        w = np.asarray(self.weights or [1.0] * len(self.members), dtype=float)
        return sum(wi * m.predict_proba(X) for wi, m in zip(w, self.members)) / w.sum()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]
