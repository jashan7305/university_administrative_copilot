"""Model zoo for the intent baseline, built on a leakage-safe preprocessing transformer."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from logic.intent.preprocess import DEFAULT, PreprocessConfig, Preprocessor, SpellCorrector


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """sklearn wrapper: the spelling vocabulary is fitted on the training fold only."""

    def __init__(self, config: PreprocessConfig = DEFAULT):
        self.config = config

    def fit(self, X, y=None):
        speller = None
        if self.config.spell_correct:
            base = Preprocessor(PreprocessConfig(spell_correct=False, lemmatize=False, remove_stopwords=False))
            speller = SpellCorrector(base.transform(X))
        self.preprocessor_ = Preprocessor(self.config, speller)
        return self

    def transform(self, X):
        return self.preprocessor_.transform(X)


def word_tfidf(**kw) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, token_pattern=r"(?u)<?\b\w+\b>?", **kw)


def word_char_tfidf() -> FeatureUnion:
    return FeatureUnion([
        ("word", word_tfidf()),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])


def make_model(name: str, config: PreprocessConfig = DEFAULT, C: float = 10.0) -> Pipeline:
    pre = ("prep", TextPreprocessor(config))
    if name == "majority":
        return Pipeline([pre, ("vec", word_tfidf()), ("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "complement_nb":
        return Pipeline([pre, ("vec", word_tfidf()), ("clf", ComplementNB(alpha=0.3))])
    if name == "logreg_word":
        return Pipeline([pre, ("vec", word_tfidf()), ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
    if name == "linear_svm":
        return Pipeline([pre, ("vec", word_char_tfidf()), ("clf", LinearSVC(C=C / 10, class_weight="balanced"))])
    if name == "logreg_word_char":
        return Pipeline([pre, ("vec", word_char_tfidf()), ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
    if name == "logreg_word_char_unweighted":
        return Pipeline([pre, ("vec", word_char_tfidf()), ("clf", LogisticRegression(C=C, max_iter=3000))])
    raise ValueError(name)


MODEL_NAMES = ["majority", "complement_nb", "logreg_word", "linear_svm", "logreg_word_char_unweighted", "logreg_word_char"]


def predict_with_rejection(model: Pipeline, texts, threshold: float, oos_label: str = "out_of_scope") -> np.ndarray:
    """Route low-confidence predictions to out_of_scope (open-set rejection)."""
    proba = model.predict_proba(texts)
    labels = model.classes_[proba.argmax(1)]
    return np.where(proba.max(1) < threshold, oos_label, labels)
