"""Configurable text preprocessing for administrative queries.

The pipeline runs in a fixed order; every step can be switched off through PreprocessConfig so the
ablation study in experiments.py can measure each step's contribution:

    unicode normalise -> lowercase -> contractions -> chat-speak / domain abbreviations
    -> entity masking -> punctuation cleanup -> tokenise -> spelling correction
    -> stopword removal -> lemmatisation or stemming
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import nltk
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer, WordNetLemmatizer

for _pkg in ("stopwords", "wordnet", "omw-1.4", "words"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:  # first run on a new machine
        nltk.download(_pkg, quiet=True)

# --------------------------------------------------------------------------- lexical resources

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "couldn't": "could not", "shouldn't": "should not",
    "wouldn't": "would not", "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "it's": "it is", "that's": "that is", "what's": "what is", "where's": "where is", "who's": "who is",
    "there's": "there is", "let's": "let us", "you're": "you are", "they're": "they are", "we're": "we are",
}

# Chat-speak and institution-specific abbreviations, expanded to the canonical form used in training.
ABBREVIATIONS = {
    "pls": "please", "plz": "please", "u": "you", "ur": "your", "r": "are", "thx": "thanks", "ty": "thank you",
    "abt": "about", "wat": "what", "hw": "how", "wht": "what", "wen": "when", "whr": "where", "reg": "registration",
    "sem": "semester", "sems": "semesters", "yr": "year", "dept": "department", "prof": "professor",
    "attnd": "attendance", "att": "attendance", "hstl": "hostel",
    "noc": "no objection certificate", "idcard": "id card", "tc": "transfer certificate",
    "lc": "leaving certificate", "fy": "first year", "sy": "second year",
    "btech": "b tech", "mtech": "m tech", "reval": "revaluation", "kt": "backlog",
    "atkt": "backlog", "dd": "demand draft", "asap": "as soon as possible", "info": "information",
    "docs": "documents", "doc": "document", "cert": "certificate", "certi": "certificate",
}

# Wh-words, negations and auxiliaries carry intent signal in short queries, so they survive stopword removal.
KEEP_WORDS = {
    "how", "what", "when", "where", "who", "whom", "which", "why", "not", "no", "nor", "can", "cannot",
    "should", "will", "do", "does", "did", "am", "is", "are", "was", "all", "before", "after", "again",
    "off", "out", "under", "over", "same", "only", "more", "most",
}

MASKS = [
    ("<email>", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("<url>", re.compile(r"\bhttps?://\S+|\bwww\.\S+")),
    ("<phone>", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("<date>", re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b")),
    ("<money>", re.compile(r"(?:rs\.?|inr|₹)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s?(?:rupees|rs)\b")),
    ("<percent>", re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent\b|per cent\b)")),
    ("<rollno>", re.compile(r"\b[a-z]\d{3}\b|\b\d{2}[a-z]{2,4}\d{3,5}\b")),
    ("<num>", re.compile(r"\b\d+(?:\.\d+)?\b")),
]
MASK_TOKENS = {m for m, _ in MASKS}

_TOKEN = re.compile(r"<[a-z]+>|[a-z]+(?:'[a-z]+)?|\d+")
_NON_TEXT = re.compile(r"[^a-z0-9<>'\s]")
_SPACES = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{2,}")  # "pleaseeee" -> "pleasee"


@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    expand_contractions: bool = True
    expand_abbreviations: bool = True
    mask_entities: bool = True
    spell_correct: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = True
    stem: bool = False

    def name(self) -> str:
        return ",".join(k for k, v in self.__dict__.items() if v) or "raw"


RAW = PreprocessConfig(False, False, False, False, False, False, False, False)
DEFAULT = PreprocessConfig()


# --------------------------------------------------------------------------- spelling correction

class SpellCorrector:
    """Norvig-style corrector restricted to the in-domain vocabulary.

    A token is corrected only if it has >= 4 letters and is unknown to both the domain vocabulary and an
    English dictionary (NLTK words), so valid words outside the training data ('call') are left alone.
    Candidates come from the domain vocabulary only, preferring edit distance 1 over 2 and then frequency,
    which keeps domain terms such as 'bonafide' or 'cgpa' as correction targets.
    """

    LETTERS = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, texts: list[str] | None = None, min_count: int = 2):
        self.counts: Counter[str] = Counter()
        if texts:
            self.fit(texts, min_count)

    def fit(self, texts: list[str], min_count: int = 2) -> "SpellCorrector":
        counts = Counter(tok for t in texts for tok in _TOKEN.findall(t.lower()))
        self.counts = Counter({w: c for w, c in counts.items() if c >= min_count})
        self.correct.cache_clear()
        return self

    def _edits1(self, w: str) -> set[str]:
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        swaps = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in self.LETTERS]
        inserts = [a + c + b for a, b in splits for c in self.LETTERS]
        return set(deletes + swaps + replaces + inserts)

    @lru_cache(maxsize=50_000)
    def correct(self, word: str) -> str:
        if len(word) < 4 or word in self.counts or not word.isalpha() or word in _english():
            return word
        e1 = self._edits1(word)
        cands = [w for w in e1 if w in self.counts]
        if not cands and len(word) >= 6:
            cands = [w2 for w1 in e1 for w2 in self._edits1(w1) if w2 in self.counts]
        return max(cands, key=self.counts.__getitem__) if cands else word


@lru_cache(maxsize=1)
def _english() -> frozenset[str]:
    return frozenset(w.lower() for w in words.words())


# --------------------------------------------------------------------------- pipeline

@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    return frozenset(stopwords.words("english")) - KEEP_WORDS


_ALL_STOPWORDS = frozenset(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()


@lru_cache(maxsize=100_000)
def _lemma(tok: str) -> str:
    # WordNet mangles function words ('as' -> 'a', 'has' -> 'ha'), so those pass through untouched.
    if tok in MASK_TOKENS or tok in _ALL_STOPWORDS or len(tok) <= 2:
        return tok
    noun = _lemmatizer.lemmatize(tok, "n")
    return _lemmatizer.lemmatize(noun, "v") if noun == tok else noun


class Preprocessor:
    def __init__(self, config: PreprocessConfig = DEFAULT, speller: SpellCorrector | None = None):
        self.config = config
        self.speller = speller

    def normalise(self, text: str) -> str:
        """Character-level cleanup that every configuration shares (safe, lossless for meaning)."""
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return _SPACES.sub(" ", text).strip()

    def tokens(self, text: str) -> list[str]:
        c = self.config
        text = self.normalise(text)
        if c.lowercase:
            text = text.lower()
        text = _REPEAT.sub(r"\1\1", text)
        if c.expand_contractions:
            text = re.sub(r"[a-z]+'[a-z]+", lambda m: CONTRACTIONS.get(m.group(0), m.group(0)), text)
        if c.mask_entities:
            for tag, pattern in MASKS:
                text = pattern.sub(f" {tag} ", text)
        text = _NON_TEXT.sub(" ", text) if c.lowercase else re.sub(r"[^\w<>'\s]", " ", text)
        toks = _TOKEN.findall(text) if c.lowercase else re.findall(r"<[a-z]+>|\w+(?:'\w+)?", text)
        if c.expand_abbreviations:
            toks = [t for tok in toks for t in ABBREVIATIONS.get(tok, tok).split()]
        if c.spell_correct and self.speller is not None:
            toks = [self.speller.correct(t) for t in toks]
        if c.remove_stopwords:
            toks = [t for t in toks if t not in _stopwords()]
        if c.lemmatize:
            toks = [_lemma(t) for t in toks]
        elif c.stem:
            toks = [t if t in MASK_TOKENS else _stemmer.stem(t) for t in toks]
        return toks

    def __call__(self, text: str) -> str:
        return " ".join(self.tokens(text))

    def transform(self, texts) -> list[str]:
        return [self(t) for t in texts]


def clean_text(text: str) -> str:
    """Stateless default cleaning (no spelling model); kept for callers that need a quick normaliser."""
    return Preprocessor(DEFAULT, speller=None)(text)
