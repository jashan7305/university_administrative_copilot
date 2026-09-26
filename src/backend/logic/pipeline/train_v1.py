"""Train and evaluate pipeline v1 (intent ensemble, CRF NER, retrieval, end-to-end). Run: python -m logic.pipeline.train_v1"""
import json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from seqeval.metrics import f1_score as seq_f1
from sklearn.metrics import accuracy_score, f1_score
from logic.intent.embedding_model import EmbeddingClassifier, EnsembleClassifier
from logic.intent.models import make_model
from logic.intent.preprocess import PreprocessConfig
from logic.ner.extractor import CRFTagger, EntityExtractor, GazetteerTagger, load_annotations, spans_to_bio, tokenize
from logic.pipeline.copilot import INTENT_PATH, NER_PATH, Copilot
from logic.retrieval.search import BM25Retriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "week3"; OUT.mkdir(parents=True, exist_ok=True)
R = {}
d = {s: pd.read_csv(ROOT / f"data/processed/{s}.csv") for s in ["train", "val", "test", "test_noisy", "test_seed", "test_clinc_oos"]}
cfg = PreprocessConfig(remove_stopwords=True)
mf = lambda y, p: round(f1_score(y, p, average="macro"), 4)

# 1. intent
cands = {"tfidf_logreg (week 2)": make_model("logreg_word_char", cfg), "minilm_logreg": EmbeddingClassifier("minilm"),
         "bge_small_logreg": EmbeddingClassifier("bge-small"),
         "ensemble_tfidf_minilm": EnsembleClassifier([make_model("logreg_word_char", cfg), EmbeddingClassifier("minilm")])}
R["intent"] = {}
for n, m in cands.items():
    m.fit(d["train"].text, d["train"].intent)
    R["intent"][n] = {s: mf(d[s].intent, m.predict(d[s].text)) for s in ["val", "test", "test_noisy", "test_seed"]}
    R["intent"][n]["clinc_oos_recall"] = round(float((m.predict(d["test_clinc_oos"].text) == "out_of_scope").mean()), 4)
    print(n, R["intent"][n])
best = max(R["intent"], key=lambda n: R["intent"][n]["val"]); R["intent_selected"] = best
joblib.dump({"model": cands[best], "threshold": 0.25}, INTENT_PATH)

# 2. NER
a = load_annotations(ROOT / "data/processed/entity_annotations.jsonl")
def ev(t, recs):
    Y, P = [], []
    for r in recs:
        tk = tokenize(r["text"]); Y.append(spans_to_bio(tk, r["entities"])); P.append(t.tag(tk))
    return round(seq_f1(Y, P), 4)
R["ner"] = {}
taggers = {"gazetteer": GazetteerTagger().fit(a["train"]), "crf": CRFTagger("none").fit(a["train"]),
           "crf+gazetteer (full)": CRFTagger("full").fit(a["train"]), "crf+gazetteer (value crossfit, 5)": CRFTagger("crossfit", n_folds=5).fit(a["train"])}
for n, t in taggers.items():
    R["ner"][n] = {"val": ev(t, a["val"]), "test": ev(t, a["test"])}; print(n, R["ner"][n])
nbest = max((n for n in taggers if n != "gazetteer"), key=lambda n: R["ner"][n]["val"]); R["ner_selected"] = nbest
EntityExtractor(taggers[nbest]).save(NER_PATH)

# 3. retrieval
qs = [json.loads(l) for l in open(ROOT / "data/eval/retrieval_queries.jsonl")]
b = BM25Retriever(); dn = DenseRetriever(); h = HybridRetriever(b, dn)
R["retrieval"] = {}
for r in (b, dn, h):
    ranks = np.array([[x.doc_id for x in r.search(q["query"], k=27)].index(q["doc_id"]) + 1 for q in qs])
    R["retrieval"][r.name] = {"R@1": round((ranks <= 1).mean(), 4), "R@3": round((ranks <= 3).mean(), 4), "MRR": round((1 / ranks).mean(), 4)}
    print(r.name, R["retrieval"][r.name])

# 4. end to end
cp = Copilot(joblib.load(INTENT_PATH), EntityExtractor.load(NER_PATH), h)
rows = []
for q in qs:
    o = cp.run(q["query"])
    rows.append({"intent_ok": o.intent == q["intent"], "proc_ok": (o.service_request or {}).get("procedure_id") == q["doc_id"],
                 "in_sources": q["doc_id"] in [s["doc_id"] for s in o.sources], "status": (o.service_request or {}).get("status", "oos"),
                 "total_ms": o.timings_ms["total"]})
e = pd.DataFrame(rows)
oos = d["test_clinc_oos"].sample(200, random_state=0).text
R["e2e"] = {"n": len(e), "intent_acc": round(e.intent_ok.mean(), 4), "procedure_acc": round(e.proc_ok.mean(), 4),
            "doc_in_top3": round(e.in_sources.mean(), 4), "status_counts": e.status.value_counts().to_dict(),
            "oos_declined_200": round(float(np.mean([cp.run(t).intent == "out_of_scope" for t in oos])), 4),
            "latency_ms_median": round(e.total_ms.median(), 1), "latency_ms_p95": round(e.total_ms.quantile(.95), 1)}
print(R["e2e"])
ex = cp.run("I lost my ID card yesterday, roll no J054. What should I do? It is urgent")
R["example"] = {"answer": ex.answer, "service_request": ex.service_request, "entities": ex.entities}
(OUT / "results.json").write_text(json.dumps(R, indent=2, default=str))
