"""Week 5: evaluation and error analysis of the copilot (final Week 4 system v3, with Week 2 and Week 3 as references).

  A. test sets       every test set used, with sizes (incl. the new challenge set (from official NMIMS sources))
  B. quantitative    metrics with bootstrap 95% CIs, McNemar significance, per-class scores, confusion matrix,
                     calibration (ECE, reliability diagram), selective prediction (risk-coverage), NER per entity
                     type, retrieval, end-to-end latency
  C. robustness      CheckList-style perturbations of held-out questions (typos, casing, abbreviations, Hinglish,
                     filler, word dropout, shuffling, truncation) + challenge-set categories + near-domain OOS
  D. error analysis  automatic error taxonomy for routing, NER and retrieval; error rate by slice
  E. failure cases   candidate failure cases per component -> reports/week5/failure_cases.json
                     (the ones discussed in the report, with root cause and fix, are in failure_notes.yaml)
  F. fixes           error-driven fixes measured before/after: typo normalisation, confidence-based abstention

Inference only (no training). Run:  python -m nlp.pipeline.evaluate_week5 [--skip-llm]
Outputs: reports/week5/results.json, errors_v3.csv, failure_cases.json, figures/
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from dataset.config import SUB_TO_INTENT, TAXONOMY
from dataset.phase3_expand import ABBREV, GREETINGS, HINGLISH, casual, typo
from nlp import DATA, MODELS, REPORTS
from nlp.intent.experiments import load as load_intent_splits
from nlp.knowledge.store import load_chunks, load_services
from nlp.pipeline.compare_week4 import NUM, _in_scope_argmax
from nlp.pipeline.copilot import Copilot
from nlp.pipeline.summarize import LLMSummarizer, MMRSummarizer
from nlp.pipeline.train_eval import queries, retrieval_eval_set
from nlp.retrieval.search import get_encoder

OUT = REPORTS / "week5"
FIG = OUT / "figures"
R: dict = {}
SEED = 42
INTENTS = list(TAXONOMY) + ["out_of_scope"]


# --------------------------------------------------------------------------- helpers

@dataclass
class System:
    """Batch view of one copilot's classifiers: intent (with OOS threshold) and sub-intent routing."""
    name: str
    intent_model: object
    threshold: float
    sub_model: object
    normalize: object = None          # optional text normaliser applied first (section F)

    def _t(self, texts):
        texts = list(texts)
        return [self.normalize(t) for t in texts] if self.normalize else texts

    def intents(self, texts) -> tuple[np.ndarray, np.ndarray]:
        p = self.intent_model.predict_proba(self._t(texts))
        conf, pred = p.max(1), self.intent_model.classes_[p.argmax(1)].astype(object)
        pred[conf < self.threshold] = "out_of_scope"
        return pred, conf

    def subs(self, texts) -> tuple[np.ndarray, np.ndarray]:
        p = self.sub_model.predict_proba(self._t(texts))
        mask = np.array([c != "out_of_scope" for c in self.sub_model.classes_])
        p = np.where(mask, p, -1)
        return self.sub_model.classes_[p.argmax(1)], p.max(1)


def bootstrap(values: np.ndarray, n: int = 1000) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean of per-example scores (0/1 correctness, reciprocal ranks, ...)."""
    rng = np.random.default_rng(SEED)
    v = np.asarray(values, dtype=float)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)


def bootstrap_f1(y, p, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    s = [f1_score(y[i], p[i], average="macro", zero_division=0) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)


def mcnemar(a_correct, b_correct) -> dict:
    """Exact McNemar test on paired correctness: only the discordant pairs matter."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": round(float(p), 5)}


def ece(conf, correct, bins: int = 10) -> float:
    """Expected calibration error: |accuracy - confidence| averaged over confidence bins, weighted by bin size."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return round(float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                           for b in range(bins) if (idx == b).any())), 4)


def risk_coverage(conf, correct) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-np.asarray(conf))
    c = np.asarray(correct, float)[order]
    acc_at = np.cumsum(c) / np.arange(1, len(c) + 1)
    cov = np.arange(1, len(c) + 1) / len(c)
    return cov, acc_at, round(float((1 - acc_at).mean()), 4)       # AURC: area under the risk-coverage curve


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160)
    plt.close()


def r4(x) -> float:
    return round(float(x), 4)


# --------------------------------------------------------------------------- perturbations (section C)

def _typos(text: str, k: int, rng) -> str:
    for _ in range(k):
        text = typo(text, rng) or text
    return text


def _drop(text: str, rng) -> str:
    w = text.split()
    if len(w) < 4:
        return text
    drop = set(rng.sample(range(len(w)), k=max(1, len(w) // 4)))
    return " ".join(x for i, x in enumerate(w) if i not in drop)


def _shuffle(text: str, rng) -> str:
    w = text.split()
    rng.shuffle(w)
    return " ".join(w)


def _abbrev_all(text: str, rng) -> str:
    return " ".join(ABBREV.get(re.sub(r"[^\w-]", "", w.lower()), w) for w in text.split())


PERTURBATIONS = {
    "typo x1": lambda t, r: _typos(t, 1, r),
    "typo x2": lambda t, r: _typos(t, 2, r),
    "typo x3": lambda t, r: _typos(t, 3, r),
    "lowercase, no punctuation": lambda t, r: casual(t, r) or t,
    "all abbreviations": _abbrev_all,
    "Hinglish suffix": lambda t, r: re.sub(r"[?.!]+$", "", t) + r.choice(HINGLISH),
    "greeting prefix": lambda t, r: r.choice(GREETINGS) + t[0].lower() + t[1:],
    "drop 25% of words": _drop,
    "shuffle word order": _shuffle,
    "first half only": lambda t, r: " ".join(t.split()[: max(1, (len(t.split()) + 1) // 2)]),
}


def perturb(name: str, text: str) -> str:
    return PERTURBATIONS[name](text, random.Random(f"{name}|{text}"))      # deterministic per (perturbation, query)


# --------------------------------------------------------------------------- typo normaliser (section F)

class SpellNormalizer:
    """Maps out-of-vocabulary words to the closest frequent training word (edit-similarity >= cutoff).
    Vocabulary = words seen at least `min_count` times in the training questions, so rare typos in the
    training data itself are not used as corrections."""

    def __init__(self, texts, min_count: int = 3, cutoff: float = 0.8):
        counts = Counter(w for t in texts for w in re.findall(r"[a-z]+", t.lower()))
        self.freq = {w: c for w, c in counts.items() if c >= min_count}
        self.words, self.cutoff = sorted(self.freq), cutoff
        self.cache: dict[str, str] = {}

    def fix(self, word: str) -> str:
        lw = word.lower()
        if len(lw) < 4 or lw in self.freq:
            return word
        if lw not in self.cache:
            cand = difflib.get_close_matches(lw, self.words, n=3, cutoff=self.cutoff)
            self.cache[lw] = max(cand, key=lambda c: self.freq[c]) if cand else lw
        return self.cache[lw]

    def __call__(self, text: str) -> str:
        return re.sub(r"[A-Za-z]+", lambda m: self.fix(m.group()), text)


# --------------------------------------------------------------------------- data

def load_challenge() -> pd.DataFrame:
    rows = yaml.safe_load((DATA / "annotations" / "challenge.yaml").read_text())["challenge"]
    as_list = lambda v: [] if v is None else (v if isinstance(v, list) else [v])
    return pd.DataFrame({"text": [r["q"] for r in rows], "intents": [as_list(r["intent"]) for r in rows],
                         "subs": [as_list(r["sub"]) for r in rows], "cat": [r["cat"] for r in rows]})


def load_gold_ner(q) -> list[dict]:
    gold_doc = yaml.safe_load((DATA / "annotations" / "ner_gold.yaml").read_text())["gold"]
    allq = pd.concat(q.values()).set_index("query_id")
    out = []
    for qid, spans in gold_doc.items():
        text = allq.loc[qid, "query"]
        out.append({"id": qid, "text": text, "variant": bool(allq.loc[qid, "is_variant"]),
                    "spans": [(text.index(str(s)), text.index(str(s)) + len(str(s)), t) for s, t in spans]})
    return out


# --------------------------------------------------------------------------- A. test sets

def test_sets(d, q, ch, gold, ev) -> None:
    print("[A] test sets")
    ins = lambda df: df[df["intent"] != "out_of_scope"]
    R["test_sets"] = [
        {"set": "test_core", "n": len(d["test_core"]), "in_scope": len(ins(d["test_core"])),
         "what": "held-out questions from official NMIMS sources (+ CLINC150 out-of-scope test queries); no group overlaps training"},
        {"set": "test_variants", "n": len(d["test_variants"]),
         "what": "rule-generated rewordings of held-out questions (typos, abbreviations, polite/formal, Hinglish)"},
        {"set": "test_clinc_oos", "n": len(d["test_clinc_oos"]), "what": "real out-of-scope user queries from CLINC150"},
        {"set": "challenge (new)", "n": len(ch), "categories": ch["cat"].value_counts().to_dict(),
         "what": "hard queries from official NMIMS sources: near-domain OOS, very short, heavy typos, code-mixed, negation, multi-intent, ambiguous, long"},
        {"set": "gold NER", "n": len(gold), "spans": sum(len(g["spans"]) for g in gold),
         "what": "hand-annotated entity spans (half core questions, half rewordings)"},
        {"set": "retrieval", "n": len(ev), "what": "verified test questions with the passages that cite their facts"},
        {"set": "robustness", "n": len(ins(d["test_core"])) * len(PERTURBATIONS),
         "what": f"{len(PERTURBATIONS)} controlled perturbations of every in-scope test_core question"},
    ]
    R["coverage"] = {"intents_in_test_core": int(ins(d["test_core"])["intent"].nunique()),
                     "sub_intents_in_test_core": int(ins(d["test_core"])["sub_intent"].nunique()), "sub_intents_total": len(SUB_TO_INTENT)}


# --------------------------------------------------------------------------- B. quantitative evaluation

def quantitative(d, systems: dict[str, System]) -> dict:
    print("[B] quantitative evaluation")
    t = d["test_core"]
    ins = t[t["intent"] != "out_of_scope"]
    rows, preds = [], {}
    for name, s in systems.items():
        pi, ci = s.intents(t["text"])
        ps, cs = s.subs(ins["text"])
        preds[name] = {"intent": pi, "intent_conf": ci, "sub": ps, "sub_conf": cs}
        ok_i, ok_s = pi == t["intent"].to_numpy(), ps == ins["sub_intent"].to_numpy()
        oos = d["test_clinc_oos"]
        rows.append({"system": name, "intent_acc": r4(ok_i.mean()), "intent_acc_ci": bootstrap(ok_i),
                     "intent_macro_f1": r4(f1_score(t["intent"], pi, average="macro")), "intent_macro_f1_ci": bootstrap_f1(t["intent"], pi),
                     "sub_acc": r4(ok_s.mean()), "sub_acc_ci": bootstrap(ok_s),
                     "sub_macro_f1": r4(f1_score(ins["sub_intent"], ps, average="macro", zero_division=0)),
                     "oos_recall_clinc": r4((s.intents(oos["text"])[0] == "out_of_scope").mean()),
                     "ece_intent": ece(ci, t["intent"].to_numpy() == s.intent_model.classes_[s.intent_model.predict_proba(s._t(t["text"])).argmax(1)]),
                     "ece_sub": ece(cs, ok_s)})
        print("   ", {k: v for k, v in rows[-1].items() if "ci" not in k})
    names = list(systems)
    v2, v3 = names[1], names[2]
    sig = {"intent": mcnemar(preds[v2]["intent"] == t["intent"].to_numpy(), preds[v3]["intent"] == t["intent"].to_numpy()),
           "sub_intent": mcnemar(preds[v2]["sub"] == ins["sub_intent"].to_numpy(), preds[v3]["sub"] == ins["sub_intent"].to_numpy()),
           "sub_intent_vs_week2": mcnemar(preds[names[0]]["sub"] == ins["sub_intent"].to_numpy(), preds[v3]["sub"] == ins["sub_intent"].to_numpy())}

    # per-intent and per-sub-intent scores of the final system
    p, r, f, n = precision_recall_fscore_support(t["intent"], preds[v3]["intent"], labels=INTENTS, zero_division=0)
    per_intent = [{"intent": c, "precision": r4(a), "recall": r4(b), "f1": r4(e), "support": int(m)} for c, a, b, e, m in zip(INTENTS, p, r, f, n)]
    subs = sorted(ins["sub_intent"].unique())
    p, r, f, n = precision_recall_fscore_support(ins["sub_intent"], preds[v3]["sub"], labels=subs, zero_division=0)
    per_sub = sorted([{"sub_intent": c, "precision": r4(a), "recall": r4(b), "f1": r4(e), "support": int(m)}
                      for c, a, b, e, m in zip(subs, p, r, f, n)], key=lambda x: (x["f1"], -x["support"]))
    R["quantitative"] = {"test_set": "test_core", "n": len(t), "n_in_scope": len(ins), "comparison": rows, "significance_v3_vs_v2": sig,
                         "per_intent_v3": per_intent, "worst_sub_intents_v3": per_sub[:10],
                         "sub_intents_f1_below_0.5": sum(x["f1"] < 0.5 for x in per_sub), "sub_intents_perfect": sum(x["f1"] == 1 for x in per_sub)}

    # confusion matrix (row-normalised) of the final system
    cm = confusion_matrix(t["intent"], preds[v3]["intent"], labels=INTENTS).astype(float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    short = [c.replace("ACADEMIC_REGISTRATION", "ACAD_REG").replace("STUDENT_", "STU_").replace("out_of_scope", "OOS") for c in INTENTS]
    ax.set_xticks(range(len(INTENTS)), short, rotation=60, ha="right", fontsize=7), ax.set_yticks(range(len(INTENTS)), short, fontsize=7)
    for i in range(len(INTENTS)):
        for j in range(len(INTENTS)):
            if cm[i, j] >= 0.02:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"), ax.set_ylabel("gold"), ax.set_title("Intent confusion matrix, final system (test_core, row-normalised)")
    savefig("confusion_intent_v3.png")

    # calibration: reliability diagram + risk-coverage (routing confidence)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    edges = np.linspace(0, 1, 11)
    for name in names:
        cs, ok = preds[name]["sub_conf"], preds[name]["sub"] == ins["sub_intent"].to_numpy()
        idx = np.clip(np.digitize(cs, edges[1:-1]), 0, 9)
        xs = [cs[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        ys = [ok[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        axes[0].plot(xs, ys, "o-", label=f"{name} (ECE {ece(cs, ok):.3f})")
        cov, acc_at, aurc = risk_coverage(cs, ok)
        axes[1].plot(cov, acc_at, label=f"{name} (AURC {aurc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8), axes[0].set_xlabel("confidence"), axes[0].set_ylabel("accuracy")
    axes[0].set_title("Reliability of routing confidence"), axes[0].legend(fontsize=7)
    axes[1].set_xlabel("coverage (share of queries answered automatically)"), axes[1].set_ylabel("routing accuracy")
    axes[1].set_title("Selective prediction: accuracy vs coverage"), axes[1].legend(fontsize=7), axes[1].set_ylim(0.5, 1.01)
    savefig("calibration_selective.png")
    sel = {}
    for name in names:
        cov, acc_at, aurc = risk_coverage(preds[name]["sub_conf"], preds[name]["sub"] == ins["sub_intent"].to_numpy())
        sel[name] = {"aurc": aurc, **{f"acc_at_{int(c * 100)}pct_coverage": r4(acc_at[int(c * len(acc_at)) - 1]) for c in (0.5, 0.7, 0.9, 1.0)}}
    R["quantitative"]["selective"] = sel

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    df = pd.DataFrame(rows).set_index("system")[["intent_acc", "sub_acc"]]
    lo = np.array([[x["intent_acc"] - x["intent_acc_ci"][0], x["sub_acc"] - x["sub_acc_ci"][0]] for x in rows]).T
    hi = np.array([[x["intent_acc_ci"][1] - x["intent_acc"], x["sub_acc_ci"][1] - x["sub_acc"]] for x in rows]).T
    xpos = np.arange(len(df))
    for j, col in enumerate(df.columns):
        ax.barh(xpos + j * 0.38, df[col], 0.38, xerr=[lo[j], hi[j]], capsize=3, label=col)
        for x, v in zip(xpos, df[col]):
            ax.text(v + 0.02, x + j * 0.38, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(xpos + 0.19, df.index, fontsize=8), ax.set_xlim(0.5, 1), ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Accuracy with 95% bootstrap confidence intervals (test_core)")
    savefig("accuracy_ci.png")
    return preds


def ner_eval(gold, extractors: dict) -> dict:
    print("[B] NER per entity type")
    out, preds_all = {}, {}
    for name, ex in extractors.items():
        preds = [[(e.start, e.end, e.label) for e in ex.extract(g["text"])] for g in gold]
        preds_all[name] = preds
        types = sorted({s[2] for g in gold for s in g["spans"]})
        per = []
        for ty in types + ["ALL"]:
            sel = (lambda s: True) if ty == "ALL" else (lambda s, ty=ty: s[2] == ty)
            tp_s = tp_r = n_p = n_g = 0
            for P, g in zip(preds, gold):
                P, G = [p for p in P if sel(p)], [s for s in g["spans"] if sel(s)]
                n_p, n_g = n_p + len(P), n_g + len(G)
                tp_s += len(set(P) & set(G))
                tp_r += sum(any(p[2] == x[2] and p[0] < x[1] and x[0] < p[1] for x in G) for p in P)
            pr, rc = tp_r / max(n_p, 1), tp_r / max(n_g, 1)
            per.append({"type": ty, "gold": n_g, "pred": n_p, "strict_f1": r4(2 * tp_s / max(n_p + n_g, 1)),
                        "relaxed_p": r4(pr), "relaxed_r": r4(rc), "relaxed_f1": r4(2 * pr * rc / max(pr + rc, 1e-9))})
        # error taxonomy over gold spans + spurious predictions
        cats = Counter()
        for P, g in zip(preds, gold):
            for s in g["spans"]:
                if s in P:
                    cats["exact"] += 1
                elif any(p[2] == s[2] and p[0] < s[1] and s[0] < p[1] for p in P):
                    cats["boundary error"] += 1
                elif any(p[0] < s[1] and s[0] < p[1] for p in P):
                    cats["wrong type"] += 1
                else:
                    cats["missed"] += 1
            cats["spurious (no gold overlap)"] += sum(not any(p[0] < s[1] and s[0] < p[1] for s in g["spans"]) for p in P)
        out[name] = {"per_type": per, "errors": dict(cats),
                     "variants_relaxed_f1": _relaxed([p for p, g in zip(preds, gold) if g["variant"]], [g for g in gold if g["variant"]]),
                     "core_relaxed_f1": _relaxed([p for p, g in zip(preds, gold) if not g["variant"]], [g for g in gold if not g["variant"]])}
        print("   ", name, out[name]["errors"], "core", out[name]["core_relaxed_f1"], "variants", out[name]["variants_relaxed_f1"])
    R["ner"] = out
    return preds_all


def _relaxed(preds, gold) -> float:
    tp = n_p = n_g = 0
    for P, g in zip(preds, gold):
        n_p, n_g = n_p + len(P), n_g + len(g["spans"])
        tp += sum(any(p[2] == s[2] and p[0] < s[1] and s[0] < p[1] for s in g["spans"]) for p in P)
    return r4(2 * tp / max(n_p + n_g, 1))


def retrieval_eval(ev, systems: dict[str, System], retrievers: dict) -> dict:
    print("[B] retrieval")
    out, ranks_all = {}, {}
    for name, s in systems.items():
        if name not in retrievers:
            continue
        pi, _ = s.intents(ev["query"])
        ps, _ = s.subs(ev["query"])
        ranks = np.array([next((r for r, h in enumerate(retrievers[name].search(x, k=20, intent=i, sub_intent=sb), 1) if h.doc_id in rel), 999)
                          for x, rel, i, sb in zip(ev["query"], ev["relevant"], pi, ps)])
        ranks_all[name] = (ranks, ps)
        rr = np.where(ranks < 999, 1 / ranks, 0)
        out[name] = {"recall@1": r4((ranks <= 1).mean()), "recall@3": r4((ranks <= 3).mean()), "recall@3_ci": bootstrap(ranks <= 3),
                     "recall@5": r4((ranks <= 5).mean()), "recall@10": r4((ranks <= 10).mean()), "mrr@20": r4(rr.mean()), "mrr_ci": bootstrap(rr)}
        print("   ", name, out[name])
    R["retrieval"] = {"n": len(ev), "comparison": out}
    return ranks_all


def end_to_end(t, copilots: dict) -> list:
    print("[B] end to end (latency, statuses)")
    out, runs = {}, []
    for name, cp in copilots.items():
        lat, status, rows = [], Counter(), []
        for r in t.itertuples():
            o = cp.run(r.query)
            lat.append(o.timings_ms["total"])
            sr = o.service_request or {}
            status[sr.get("status", "out_of_scope")] += 1
            rows.append(o)
        out[name] = {"latency_ms_p50": round(float(np.percentile(lat, 50)), 1), "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
                     "status": dict(status)}
        runs.append(rows)
        print("   ", name, out[name])
    R["end_to_end"] = {"n": len(t), **out}
    return runs[-1]


# --------------------------------------------------------------------------- C. robustness

def robustness(d, ch, systems: dict[str, System], ev, retrievers) -> None:
    print("[C] robustness")
    ins = d["test_core"][d["test_core"]["intent"] != "out_of_scope"]
    gold_sub = ins["sub_intent"].to_numpy()
    rows = []
    clean = {n: s.subs(ins["text"])[0] for n, s in systems.items()}
    for n in systems:
        rows.append({"perturbation": "clean", "system": n, "sub_acc": r4((clean[n] == gold_sub).mean()), "consistency": 1.0, "changed": 0.0})
    for pname in PERTURBATIONS:
        texts = [perturb(pname, x) for x in ins["text"]]
        changed = np.mean([a != b for a, b in zip(texts, ins["text"])])
        for n, s in systems.items():
            p = s.subs(texts)[0]
            rows.append({"perturbation": pname, "system": n, "sub_acc": r4((p == gold_sub).mean()),
                         "consistency": r4((p == clean[n]).mean()), "changed": r4(changed)})
        print("   ", pname, [(r["system"], r["sub_acc"]) for r in rows[-len(systems):]])
    df = pd.DataFrame(rows)
    piv = df.pivot(index="perturbation", columns="system", values="sub_acc").loc[["clean", *PERTURBATIONS]][list(systems)]
    ax = piv.plot.barh(figsize=(8, 6.5), width=0.8)
    ax.invert_yaxis(), ax.set_xlim(0, 1), ax.set_xlabel("routing (sub-intent) accuracy"), ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Robustness: routing accuracy under controlled perturbations (test_core)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=6, padding=2)
    savefig("robustness.png")
    clean_acc = {n: rows[i]["sub_acc"] for i, n in enumerate(systems)}
    worst = {n: min((r for r in rows if r["system"] == n and r["perturbation"] != "clean"), key=lambda r: r["sub_acc"]) for n in systems}

    # retrieval under typos
    ret = {}
    for pname in ("typo x2", "lowercase, no punctuation"):
        texts = [perturb(pname, x) for x in ev["query"]]
        for n, s in systems.items():
            if n not in retrievers:
                continue
            pi, ps = s.intents(texts)[0], s.subs(texts)[0]
            hit = [any(h.doc_id in rel for h in retrievers[n].search(x, k=3, intent=i, sub_intent=sb))
                   for x, rel, i, sb in zip(texts, ev["relevant"], pi, ps)]
            ret[f"{n} | {pname}"] = r4(np.mean(hit))
    # challenge set by category
    cats = []
    for n, s in systems.items():
        pi, ps = s.intents(ch["text"])[0], s.subs(ch["text"])[0]
        ok_i = np.array([p in g for p, g in zip(pi, ch["intents"])])
        ok_s = np.array([(not g) or (p in g) for p, g in zip(ps, ch["subs"])]) & ok_i
        for c in ch["cat"].unique():
            m = (ch["cat"] == c).to_numpy()
            cats.append({"system": n, "category": c, "n": int(m.sum()), "intent_acc": r4(ok_i[m].mean()), "intent_and_sub_acc": r4(ok_s[m].mean())})
        cats.append({"system": n, "category": "ALL", "n": len(ch), "intent_acc": r4(ok_i.mean()), "intent_and_sub_acc": r4(ok_s.mean())})
    cdf = pd.DataFrame(cats)
    piv = cdf[cdf["category"] != "ALL"].pivot(index="category", columns="system", values="intent_acc")[list(systems)]
    ax = piv.plot.barh(figsize=(8, 5.5), width=0.8)
    ax.set_xlim(0, 1), ax.set_xlabel("intent accuracy (any acceptable label)"), ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Challenge set: accuracy by stress category")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=6, padding=2)
    savefig("challenge.png")
    print("   challenge", cdf[cdf["category"] == "ALL"].to_dict("records"))
    R["robustness"] = {"perturbations": rows, "clean_acc": clean_acc, "worst_case": worst, "retrieval_recall@3_perturbed": ret,
                       "challenge": cats}


# --------------------------------------------------------------------------- D. error analysis

def error_analysis(d, q, preds, v3name) -> pd.DataFrame:
    print("[D] error analysis")
    t = d["test"].copy()
    meta = q["test"].drop_duplicates("query").set_index("query")
    s = preds["system"]
    t["pred_intent"], t["conf"] = s.intents(t["text"])
    t["pred_sub"], t["sub_conf"] = s.subs(t["text"])
    t["variant_type"] = t["text"].map(meta["variant_type"]).fillna("clinc150")
    t["alternatives"] = t["text"].map(meta["alternative_sub_intents"]).fillna("")
    t["ambiguity"] = t["text"].map(meta["ambiguity"]).fillna("")

    def category(r) -> str:
        if r.intent == "out_of_scope":
            return "missed out-of-scope" if r.pred_intent != "out_of_scope" else "ok"
        if r.pred_intent == "out_of_scope":
            return "false out-of-scope"
        if r.pred_intent != r.intent:
            return "wrong intent"
        if r.pred_sub != r.sub_intent:
            return "wrong sub-intent, same intent" if SUB_TO_INTENT.get(r.pred_sub) == r.intent else "sub-intent outside predicted intent"
        return "ok"
    t["error"] = [category(r) for r in t.itertuples()]
    t["plausible_alternative"] = [r.pred_sub in str(r.alternatives).split("|") for r in t.itertuples()]
    t["n_words"] = t["text"].str.split().str.len()
    err = t[t["error"] != "ok"]
    err[["text", "intent", "sub_intent", "pred_intent", "pred_sub", "conf", "sub_conf", "error", "variant_type", "plausible_alternative",
         "ambiguity"]].sort_values(["error", "conf"], ascending=[True, False]).to_csv(OUT / "errors_v3.csv", index=False)

    ins = t[t["intent"] != "out_of_scope"].copy()
    ins["wrong"] = ins["error"] != "ok"
    bucket = lambda df, col: [{"slice": str(k), "n": int(len(g)), "error_rate": r4(g["wrong"].mean())} for k, g in df.groupby(col, observed=True)]
    ins["length"] = pd.cut(ins["n_words"], [0, 5, 10, 15, 25, 200], labels=["1-5 words", "6-10", "11-15", "16-25", "26+"])
    ins["confidence"] = pd.cut(ins["sub_conf"], [0, 0.3, 0.5, 0.7, 0.9, 1.01], labels=["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">=0.9"])
    slices = {"by_variant_type": bucket(ins, "variant_type"), "by_length": bucket(ins, "length"), "by_confidence": bucket(ins, "confidence"),
              "by_intent": sorted(bucket(ins, "intent"), key=lambda x: -x["error_rate"])}
    pairs = Counter(zip(err[err["error"].str.startswith(("wrong sub", "sub-intent"))]["sub_intent"],
                        err[err["error"].str.startswith(("wrong sub", "sub-intent"))]["pred_sub"]))
    tax = err["error"].value_counts().to_dict()
    R["errors"] = {"n_test": len(t), "n_errors": len(err), "taxonomy": tax,
                   "errors_with_plausible_alternative": int(err["plausible_alternative"].sum()),
                   "routing_errors": int(err["error"].str.startswith(("wrong sub", "sub-intent")).sum()),
                   "high_confidence_errors": int((err["conf"] >= 0.8).sum()),
                   "top_confused_sub_intents": [{"gold": a, "predicted": b, "count": c} for (a, b), c in pairs.most_common(10)],
                   "slices": slices}
    print("   ", tax, "plausible alt:", R["errors"]["errors_with_plausible_alternative"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    pd.Series(tax).sort_values().plot.barh(ax=axes[0], color="#c0504d")
    axes[0].set_title(f"Error types ({len(err)} errors / {len(t)} test queries)", fontsize=9)
    for ax, key, title in ((axes[1], "by_variant_type", "Routing error rate by query style"), (axes[2], "by_confidence", "Routing error rate by confidence")):
        s_ = pd.DataFrame(slices[key]).set_index("slice")["error_rate"]
        s_.plot.bar(ax=ax, color="#4f81bd")
        ax.set_title(title, fontsize=9), ax.set_ylim(0, 1), ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        for i, v in enumerate(s_):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    savefig("errors.png")
    return t


# --------------------------------------------------------------------------- summarization faithfulness (B/D) + E. failure cases

def summarization_errors(q, v3: Copilot, skip_llm: bool) -> list:
    if skip_llm:
        return []
    print("[D] LLM summarisation faithfulness")
    services = load_services()
    t = q["test"][(q["test"]["source_status"] == "VERIFIED") & ~q["test"]["is_variant"] & ~q["test"]["multi_intent"]]
    t = t.sample(min(60, len(t)), random_state=11)
    llm, mmr, enc = LLMSummarizer(), MMRSummarizer(), get_encoder()
    cases, stats = [], Counter()
    for text, ref in zip(t["query"], t["answer_facts"]):
        intent, _, _ = v3.classify(text)
        sub = v3.route(text, intent) if intent != "out_of_scope" else None
        hits = v3.retriever.search(text, k=3, intent=intent, sub_intent=sub)
        inputs = [h.chunk.text for h in hits] + list(services[sub].facts if sub in services else [])
        evidence = " ".join(inputs).lower()
        a_llm, a_mmr = llm.summarize(text, inputs), mmr.summarize(text, inputs)
        nums = [n.lower().strip(".,") for n in NUM.findall(a_llm)]
        bad = [n for n in nums if n not in evidence]
        e = enc.encode([ref, a_llm or " ", a_mmr or " "], normalize_embeddings=True, show_progress_bar=False)
        stats["answers"] += 1
        stats["with_unsupported_number"] += bool(bad)
        stats["off_topic_llm (sim<0.5)"] += float(e[0] @ e[1]) < 0.5
        stats["llm_worse_than_mmr_by_0.15"] += float(e[0] @ e[2]) - float(e[0] @ e[1]) > 0.15
        cases.append({"query": text, "reference": ref, "llm": a_llm, "mmr": a_mmr, "unsupported": bad,
                      "sim_llm": r4(e[0] @ e[1]), "sim_mmr": r4(e[0] @ e[2])})
    R["summarization_errors"] = dict(stats)
    print("   ", dict(stats))
    return cases


def failure_cases(t, gold, ner_preds, ranks_all, ev, sum_cases, ch, v3sys, v3name, v2name) -> None:
    print("[E] failure cases")
    pools: dict[str, list] = {}
    err = t[t["error"] != "ok"]
    pick = lambda df, n=6: df.head(n)
    for cat in err["error"].unique():
        pools[f"routing: {cat}"] = [
            {"query": r.text, "gold": f"{r.intent} / {r.sub_intent}", "predicted": f"{r.pred_intent} / {r.pred_sub}",
             "confidence": r4(r.conf), "sub_confidence": r4(r.sub_conf), "style": r.variant_type,
             "plausible_alternative": bool(r.plausible_alternative)}
            for r in pick(err[err["error"] == cat].sort_values("conf", ascending=False)).itertuples()]
    # challenge failures of the final system
    pi, ps = v3sys.intents(ch["text"])[0], v3sys.subs(ch["text"])[0]
    pools["challenge"] = [{"query": x, "category": c, "gold": f"{g} / {s}", "predicted": f"{p} / {sp}"}
                          for x, c, g, s, p, sp in zip(ch["text"], ch["cat"], ch["intents"], ch["subs"], pi, ps) if p not in g][:40]
    # NER
    ner = []
    for g, P in zip(gold, ner_preds[v3name]):
        missed = [g["text"][a:b] + f" [{ty}]" for a, b, ty in g["spans"] if not any(p[0] < b and a < p[1] and p[2] == ty for p in P)]
        spurious = [g["text"][a:b] + f" [{ty}]" for a, b, ty in P if not any(a < s[1] and s[0] < b for s in g["spans"])]
        wrong = [g["text"][a:b] + f" [pred {ty}]" for a, b, ty in P if any(a < s[1] and s[0] < b and s[2] != ty for s in g["spans"])]
        if missed or spurious or wrong:
            ner.append({"query": g["text"], "gold": [g["text"][a:b] + f" [{ty}]" for a, b, ty in g["spans"]],
                        "missed": missed, "spurious": spurious, "wrong_type": wrong})
    pools["ner"] = ner[:40]
    # retrieval misses of the final system, split by whether routing was right
    ranks, ps = ranks_all[v3name]
    gold_sub = ev["sub_intent"].str.split("|").str[0].to_numpy()
    miss = [{"query": x, "gold_sub": g, "pred_sub": p, "rank_of_first_relevant": int(r) if r < 999 else ">20", "routing_correct": bool(g == p)}
            for x, g, p, r in zip(ev["query"], gold_sub, ps, ranks) if r > 3]
    pools["retrieval: routing correct"] = [m for m in miss if m["routing_correct"]][:20]
    pools["retrieval: routing wrong"] = [m for m in miss if not m["routing_correct"]][:20]
    R["errors"]["retrieval_misses_top3"] = {"total": len(miss), "with_correct_routing": sum(m["routing_correct"] for m in miss)}
    pools["summarization"] = sorted([c for c in sum_cases if c["unsupported"] or c["sim_llm"] < 0.5], key=lambda c: c["sim_llm"])[:20]
    (OUT / "failure_cases.json").write_text(json.dumps(pools, indent=2, ensure_ascii=False, default=str))
    print("    pools:", {k: len(v) for k, v in pools.items()})


# --------------------------------------------------------------------------- F. error-driven fixes

def fixes(d, ch, v3sys: System, v3name: str) -> None:
    print("[F] error-driven fixes")
    norm = SpellNormalizer(d["train"]["text"])
    fixed = System(v3name + " + typo normaliser", v3sys.intent_model, v3sys.threshold, v3sys.sub_model, norm)
    ins = lambda df: df[df["intent"] != "out_of_scope"]
    sets = {"validation (clean)": ins(d["val"]), "test_core (clean)": ins(d["test_core"]),
            "test_variants": ins(d["test_variants"])}
    rows = []
    for name, df in sets.items():
        rows.append({"set": name, "before": r4((v3sys.subs(df["text"])[0] == df["sub_intent"]).mean()),
                     "after": r4((fixed.subs(df["text"])[0] == df["sub_intent"]).mean()), "n": len(df)})
    core = ins(d["test_core"])
    for p in ("typo x1", "typo x2", "typo x3"):
        texts = [perturb(p, x) for x in core["text"]]
        rows.append({"set": f"test_core, {p}", "before": r4((v3sys.subs(texts)[0] == core["sub_intent"]).mean()),
                     "after": r4((fixed.subs(texts)[0] == core["sub_intent"]).mean()), "n": len(core)})
    m = (ch["cat"] == "heavy_typos").to_numpy()
    for label, s in (("before", v3sys), ("after", fixed)):
        pi = s.intents(ch["text"][m])[0]
        acc_ = np.mean([p in g for p, g in zip(pi, ch["intents"][m])])
        if label == "before":
            rows.append({"set": "challenge heavy_typos (intent)", "before": r4(acc_), "n": int(m.sum())})
        else:
            rows[-1]["after"] = r4(acc_)
    for r in rows:
        r["delta"] = r4(r["after"] - r["before"])
        print("   ", r)
    ex = [{"before": x, "after": norm(x)} for x in ch["text"][m].head(5)]

    # abstention: route low-confidence queries to staff; threshold picked on validation for >= 90% accuracy
    vin = ins(d["val"])
    vs, vc = v3sys.subs(vin["text"])
    ok_v = vs == vin["sub_intent"].to_numpy()
    cands = sorted(set(np.round(vc, 2)))
    thr = next((c for c in cands if ok_v[vc >= c].mean() >= 0.90), cands[-1])
    ts, tc = v3sys.subs(core["text"])
    ok_t = ts == core["sub_intent"].to_numpy()
    abst = {"threshold": float(thr), "target_val_accuracy": 0.9,
            "test_coverage": r4((tc >= thr).mean()), "test_accuracy_answered": r4(ok_t[tc >= thr].mean()),
            "test_accuracy_all": r4(ok_t.mean()), "share_sent_to_staff": r4((tc < thr).mean()),
            "errors_caught": r4((~ok_t & (tc < thr)).sum() / max((~ok_t).sum(), 1))}
    print("    abstention", abst)
    R["fixes"] = {"typo_normaliser": {"vocabulary_words": len(norm.words), "rows": rows, "examples": ex}, "abstention": abst}


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d, q, ch = load_intent_splits(), queries(), load_challenge()
    gold = load_gold_ner(q)
    ev = retrieval_eval_set(q, load_chunks())
    test_sets(d, q, ch, gold, ev)

    v2, v3 = Copilot.from_disk("v2"), Copilot.from_disk("v3")
    w2i, w2s = joblib.load(MODELS / "intent_baseline.joblib"), joblib.load(MODELS / "subintent_baseline.joblib")
    names = ("Week 2 TF-IDF", "Week 3 pipeline (v2)", "Week 4 final (v3)")
    systems = {names[0]: System(names[0], w2i["model"], w2i["threshold"], w2s["model"]),
               names[1]: System(names[1], v2.intent_model, v2.threshold, v2.sub_model),
               names[2]: System(names[2], v3.intent_model, v3.threshold, v3.sub_model)}
    preds = quantitative(d, systems)
    ner_preds = ner_eval(gold, {names[1]: v2.extractor, names[2]: v3.extractor})
    retrievers = {names[1]: v2.retriever, names[2]: v3.retriever}
    ranks_all = retrieval_eval(ev, systems, retrievers)
    core_verified = q["test"][~q["test"]["is_variant"] & ~q["test"]["multi_intent"] & (q["test"]["source_status"] == "VERIFIED")]
    end_to_end(core_verified, {names[1]: v2, names[2]: v3})
    robustness(d, ch, systems, ev, retrievers)
    t = error_analysis(d, q, {"system": systems[names[2]]}, names[2])
    sum_cases = summarization_errors(q, v3, args.skip_llm)
    failure_cases(t, gold, ner_preds, ranks_all, ev, sum_cases, ch, systems[names[2]], names[2], names[1])
    fixes(d, ch, systems[names[2]], names[2])
    R["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"done in {R['runtime_s']} s")


if __name__ == "__main__":
    main()
