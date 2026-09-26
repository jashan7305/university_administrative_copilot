"""Named entity recognition for administrative queries.

Three layers, combined by EntityExtractor:
  1. RuleExtractor      - regular expressions for pattern-shaped entities (DATE, MONEY, PHONE, EMAIL,
                          STUDENT_ID, PERCENTAGE); high precision, no training needed.
  2. GazetteerTagger    - longest-match dictionary built from training spans (baseline).
  3. CRFTagger          - linear-chain CRF (sklearn-crfsuite) over BIO tags with lexical, shape,
                          context-window and gazetteer features; generalises to unseen entity values.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import sklearn_crfsuite

_TOKEN = re.compile(r"\w+(?:[-/.']\w+)*|[^\w\s]")


@dataclass
class Entity:
    label: str
    text: str
    start: int
    end: int
    source: str = "crf"

    def as_dict(self) -> dict:
        return {"label": self.label, "text": self.text, "start": self.start, "end": self.end, "source": self.source}


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def spans_to_bio(tokens, spans) -> list[str]:
    tags = ["O"] * len(tokens)
    for sp in spans:
        inside = [i for i, (_, s, e) in enumerate(tokens) if s >= sp["start"] and e <= sp["end"]]
        for j, i in enumerate(inside):
            tags[i] = ("B-" if j == 0 else "I-") + sp["label"]
    return tags


def bio_to_entities(text, tokens, tags, source="crf") -> list[Entity]:
    ents, cur = [], None
    for (tok, s, e), tag in zip(tokens, tags):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[0] != tag[2:])):
            if cur:
                ents.append(cur)
            cur = [tag[2:], s, e]
        elif tag.startswith("I-") and cur:
            cur[2] = e
        else:
            if cur:
                ents.append(cur)
            cur = None
    if cur:
        ents.append(cur)
    return [Entity(l, text[s:e], s, e, source) for l, s, e in ents]


def load_annotations(path: Path) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["split"]].append(r)
    return out


# --------------------------------------------------------------------------- rules

RULES = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("MONEY", re.compile(r"(?:\brs\.?|\binr|₹)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s?(?:rupees|/-)", re.I)),
    ("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+\d{4})?\b"
                        r"|\b(?:today|tomorrow|yesterday|next week|this week)\b", re.I)),
    ("PERCENTAGE", re.compile(r"\b\d{1,3}(?:\.\d+)?\s?(?:%|percent\b|per cent\b)", re.I)),
    ("STUDENT_ID", re.compile(r"\b[A-Za-z]\d{3}\b|\b\d{2}[A-Za-z]{2,4}\d{3,5}\b")),
]


class RuleExtractor:
    def extract(self, text: str) -> list[Entity]:
        ents = []
        for label, pat in RULES:
            ents += [Entity(label, m.group(), m.start(), m.end(), "rule") for m in pat.finditer(text)]
        return ents


# --------------------------------------------------------------------------- gazetteer baseline

class GazetteerTagger:
    def fit(self, records, exclude: set[str] | None = None) -> "GazetteerTagger":
        self.lexicon: dict[tuple[str, ...], str] = {}
        for r in records:
            for sp in r["entities"]:
                if exclude and sp["value"].lower() in exclude:
                    continue
                self.lexicon[tuple(t.lower() for t, _, _ in tokenize(sp["value"]))] = sp["label"]
        self.max_len = max(len(k) for k in self.lexicon)
        return self

    def tag(self, tokens) -> list[str]:
        words = [t.lower() for t, _, _ in tokens]
        tags, i = ["O"] * len(words), 0
        while i < len(words):
            for n in range(min(self.max_len, len(words) - i), 0, -1):
                label = self.lexicon.get(tuple(words[i:i + n]))
                if label:
                    tags[i:i + n] = ["B-" + label] + ["I-" + label] * (n - 1)
                    i += n
                    break
            else:
                i += 1
        return tags


# --------------------------------------------------------------------------- CRF

def _shape(w: str) -> str:
    s = re.sub(r"[A-Z]", "X", w)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1+", r"\1\1", s)


def token_features(words: list[str], i: int, gaz_tags: list[str] | None) -> dict:
    w = words[i]
    f = {"bias": 1.0, "w.lower": w.lower(), "w.suffix3": w[-3:].lower(), "w.suffix2": w[-2:].lower(),
         "w.prefix3": w[:3].lower(), "w.isupper": w.isupper(), "w.istitle": w.istitle(), "w.isdigit": w.isdigit(),
         "w.shape": _shape(w), "w.len": min(len(w), 12), "w.hasdigit": any(c.isdigit() for c in w)}
    if gaz_tags is not None:
        f["gaz"] = gaz_tags[i]
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(words):
            f[f"{off}:w.lower"] = words[j].lower()
            f[f"{off}:w.istitle"] = words[j].istitle()
            f[f"{off}:w.shape"] = _shape(words[j])
            if gaz_tags is not None:
                f[f"{off}:gaz"] = gaz_tags[j]
        else:
            f[f"{off}:pad"] = True
    if i > 0:
        f["bigram-1"] = f"{words[i-1].lower()}|{w.lower()}"
    if i < len(words) - 1:
        f["bigram+1"] = f"{w.lower()}|{words[i+1].lower()}"
    return f


class CRFTagger:
    """gazetteer: 'none' | 'full' | 'crossfit'.

    'full' computes training-time dictionary features from a gazetteer that contains every training value,
    so the feature is always on during training and the CRF learns to over-trust it; it then misses values
    absent from the dictionary. 'crossfit' partitions the entity *values* into folds and computes each
    training sentence's features from a gazetteer that omits one fold's values (out-of-fold, at value level:
    the same value recurs across many sentences, so sentence-level folds would not hide anything). Training
    then sees realistic dictionary coverage, and the CRF learns to use context when the dictionary is silent.
    """

    def __init__(self, gazetteer: str = "crossfit", c1: float = 0.05, c2: float = 0.05, max_iterations: int = 200,
                 n_folds: int = 2, seed: int = 42):
        self.gazetteer_mode, self.n_folds, self.seed = gazetteer, n_folds, seed
        self.params = dict(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations, all_possible_transitions=True)

    def _featurise(self, tokens, gazetteer: "GazetteerTagger | None" = None) -> list[dict]:
        words = [t for t, _, _ in tokens]
        gz = gazetteer if gazetteer is not None else getattr(self, "gazetteer", None)
        gaz = gz.tag(tokens) if self.gazetteer_mode != "none" and gz is not None else None
        return [token_features(words, i, gaz) for i in range(len(words))]

    def fit(self, records) -> "CRFTagger":
        import random

        self.gazetteer = GazetteerTagger().fit(records)
        fold_of = {}
        if self.gazetteer_mode == "crossfit":
            rng = random.Random(self.seed)
            values = sorted({sp["value"].lower() for r in records for sp in r["entities"]})
            rng.shuffle(values)
            value_fold = {v: n % self.n_folds for n, v in enumerate(values)}
            fold_gaz = {f: GazetteerTagger().fit(records, exclude={v for v, vf in value_fold.items() if vf == f})
                        for f in range(self.n_folds)}
            fold_of = {i: rng.randrange(self.n_folds) for i in range(len(records))}
        X, y = [], []
        for i, r in enumerate(records):
            toks = tokenize(r["text"])
            gz = fold_gaz[fold_of[i]] if self.gazetteer_mode == "crossfit" else self.gazetteer
            X.append(self._featurise(toks, gz))
            y.append(spans_to_bio(toks, r["entities"]))
        self.crf = sklearn_crfsuite.CRF(**self.params)
        self.crf.fit(X, y)
        return self

    def tag(self, tokens) -> list[str]:
        return self.crf.predict_single(self._featurise(tokens)) if tokens else []

    def extract(self, text: str) -> list[Entity]:
        toks = tokenize(text)
        return bio_to_entities(text, toks, self.tag(toks))


# --------------------------------------------------------------------------- combined

class EntityExtractor:
    """Rule entities plus CRF entities; on overlap the rule entity wins (rules are near-exact patterns)."""

    def __init__(self, crf: CRFTagger):
        self.crf, self.rules = crf, RuleExtractor()

    def extract(self, text: str) -> list[Entity]:
        rule_ents = self.rules.extract(text)
        out = list(rule_ents)
        for e in self.crf.extract(text):
            if not any(e.start < r.end and r.start < e.end for r in rule_ents):
                out.append(e)
        return sorted(out, key=lambda e: e.start)

    def save(self, path: Path) -> None:
        joblib.dump(self.crf, path)

    @classmethod
    def load(cls, path: Path) -> "EntityExtractor":
        return cls(joblib.load(path))
