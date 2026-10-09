"""Phase 6 - train and evaluate every component of NLP pipeline v2 on the NMIMS dataset.

  1 classification     intent: TF-IDF (Week 2 best) vs sentence embeddings (MiniLM, BGE) vs kNN vs ensemble;
                       sub-intent routing: TF-IDF vs embeddings vs ensemble, flat vs hierarchical
  2 NER                gazetteer vs CRF (with / without cross-fitted gazetteer) on NMIMS entity annotations
  3 semantic similarity paraphrase matching: find each reworded test question's original among all core questions
  4 RAG retrieval      BM25 vs dense vs hybrid (RRF) vs hybrid + intent boost over 727 official NMIMS passages
  5 summarization      lead vs TextRank vs MMR query-focused summaries, ROUGE + embedding similarity vs gold facts
  6 information extraction / end-to-end: full copilot on held-out core questions + real out-of-scope queries

All model selection uses train/validation only; test splits are scored once. Models are saved to models/ and all
numbers to reports/week3/results.json (+ figures). Run:  python -m nlp.pipeline.train_eval
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from nlp import DATA, MODELS, REPORTS
from nlp.intent.embedding_model import EMBEDDERS, EmbeddingClassifier, EnsembleClassifier
from nlp.intent.experiments import load as load_intent_splits
from nlp.intent.models import make_model, predict_with_rejection
from nlp.intent.preprocess import PreprocessConfig
from nlp.knowledge.store import load_chunks, load_services
from nlp.ner.extractor import CRFTagger, EntityExtractor, GazetteerTagger, spans_to_bio, tokenize
from nlp.pipeline.copilot import INTENT_PATH, NER_PATH, SUBINTENT_PATH, Copilot
from nlp.pipeline.summarize import LeadSummarizer, MMRSummarizer, TextRankSummarizer
from nlp.retrieval.search import BM25Retriever, DenseRetriever, HybridRetriever, get_encoder

OUT = REPORTS / "week3"
FIG = OUT / "figures"
CFG = PreprocessConfig(remove_stopwords=True)   # selected by the Week 2 ablation
R: dict = {}


def mf1(y, p) -> float:
    return round(f1_score(y, p, average="macro", zero_division=0), 4)


def acc(y, p) -> float:
    return round(accuracy_score(y, p), 4)


def bar(df: pd.DataFrame, x: str, cols: list[str], title: str, name: str) -> None:
    ax = df.set_index(x)[cols].plot.barh(figsize=(7, 0.45 * len(df) + 1.5), width=0.8)
    ax.set_xlim(0, 1), ax.set_title(title), ax.set_ylabel(""), ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160)
    plt.close()


def queries() -> dict[str, pd.DataFrame]:
    return {s: pd.read_csv(DATA / "final" / f"{n}.csv") for s, n in (("train", "train"), ("val", "validation"), ("test", "test"))}


# --------------------------------------------------------------------------- 1. classification

def classification(d: dict) -> tuple[object, float, object]:
    print("[1] intent classification")
    tr = d["train"]
    cands = {
        "tfidf_logreg (week 2 best)": make_model("logreg_word_char", CFG),
        "minilm_logreg": EmbeddingClassifier("minilm"),
        "bge_small_logreg": EmbeddingClassifier("bge-small"),
        "ensemble_tfidf_minilm": EnsembleClassifier([make_model("logreg_word_char", CFG), EmbeddingClassifier("minilm")]),
        "ensemble_tfidf_bge": EnsembleClassifier([make_model("logreg_word_char", CFG), EmbeddingClassifier("bge-small")]),
    }
    rows = []
    for name, m in cands.items():
        m.fit(tr["text"], tr["intent"])
        rows.append({"model": name, **{s: mf1(d[s]["intent"], m.predict(d[s]["text"])) for s in ("val", "test", "test_core", "test_variants")},
                     "clinc_oos_recall": acc(d["test_clinc_oos"]["intent"], m.predict(d["test_clinc_oos"]["text"]))})
        print("   ", rows[-1])
    # kNN over embeddings = classification by semantic similarity to labelled training questions
    enc = get_encoder(EMBEDDERS["minilm"])
    etr = enc.encode(tr["text"].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    def knn(texts, k=5):
        e = enc.encode(list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        top = np.argsort(-(e @ etr.T), axis=1)[:, :k]
        labels = tr["intent"].to_numpy()
        return np.array([pd.Series(labels[t]).mode()[0] for t in top])
    rows.append({"model": "minilm_knn (k=5)", **{s: mf1(d[s]["intent"], knn(d[s]["text"])) for s in ("val", "test", "test_core", "test_variants")},
                 "clinc_oos_recall": acc(d["test_clinc_oos"]["intent"], knn(d["test_clinc_oos"]["text"]))})
    print("   ", rows[-1])
    table = pd.DataFrame(rows)
    best = max((n for n in cands), key=lambda n: table.set_index("model").loc[n, "val"])
    model = cands[best]
    # confidence threshold for out-of-scope rejection, chosen on validation macro-F1
    sweep = [(t, mf1(d["val"]["intent"], predict_with_rejection(model, d["val"]["text"], t))) for t in np.arange(0, 0.61, 0.05)]
    thr = round(float(max(sweep, key=lambda x: (x[1], -x[0]))[0]), 2)
    final = {s: {"accuracy": acc(d[s]["intent"], p := predict_with_rejection(model, d[s]["text"], thr)), "macro_f1": mf1(d[s]["intent"], p)}
             for s in ("test", "test_core", "test_variants")}
    final["clinc_oos_recall"] = acc(d["test_clinc_oos"]["intent"], predict_with_rejection(model, d["test_clinc_oos"]["text"], thr))
    R["intent"] = {"comparison": rows, "selected": best, "threshold": thr, "final": final}
    joblib.dump({"model": model, "threshold": thr, "name": best}, INTENT_PATH)
    bar(table, "model", ["val", "test_core", "clinc_oos_recall"], "Intent classification (macro-F1 / OOS recall)", "intent_models.png")

    print("[1b] sub-intent routing")
    ins = {s: d[s][d[s]["intent"] != "out_of_scope"] for s in ("train", "val", "test", "test_core", "test_variants")}
    subs = {"tfidf_logreg": make_model("logreg_word_char", CFG), "minilm_logreg": EmbeddingClassifier("minilm"),
            "ensemble_tfidf_minilm": EnsembleClassifier([make_model("logreg_word_char", CFG), EmbeddingClassifier("minilm")])}
    services = load_services()
    srows = []
    for name, m in subs.items():
        m.fit(ins["train"]["text"], ins["train"]["sub_intent"])
        row = {"model": name}
        for s in ("val", "test_core", "test"):
            df = ins[s]
            proba = m.predict_proba(df["text"])
            flat = m.classes_[proba.argmax(1)]
            pred_int = model.predict(df["text"])
            hier = []
            for pr, it in zip(proba, pred_int):
                allowed = [i for i, c in enumerate(m.classes_) if c in services and services[c].intent == it]
                hier.append(m.classes_[max(allowed, key=lambda i: pr[i])] if allowed else m.classes_[pr.argmax()])
            row[f"{s}_flat_acc"], row[f"{s}_hier_acc"] = acc(df["sub_intent"], flat), acc(df["sub_intent"], hier)
            row[f"{s}_flat_macro_f1"] = mf1(df["sub_intent"], flat)
        srows.append(row)
        print("   ", row)
    score = lambda r: max(r["val_flat_acc"], r["val_hier_acc"])
    sbest = max(subs, key=lambda n: score(next(r for r in srows if r["model"] == n)))
    brow = next(r for r in srows if r["model"] == sbest)
    strategy = "hierarchical" if brow["val_hier_acc"] > brow["val_flat_acc"] else "flat"
    joblib.dump({"model": subs[sbest], "name": sbest, "strategy": strategy}, SUBINTENT_PATH)
    R["sub_intent"] = {"comparison": srows, "selected": sbest, "strategy": strategy,
                       "n_classes": int(ins["train"]["sub_intent"].nunique())}
    return model, thr, subs[sbest]


# --------------------------------------------------------------------------- 2. NER

def ner(q: dict) -> CRFTagger:
    print("[2] NER")
    rec = lambda df: [{"text": t, "is_variant": v, "entities": [{"start": e["start"], "end": e["end"], "label": e["type"], "value": e["text"]}
                                                                 for e in json.loads(ents)]}
                      for t, ents, v in zip(df["query"], df["entities"], df["is_variant"])]
    data = {s: rec(df) for s, df in q.items()}

    def score(tagger, recs):
        Y, P = [], []
        for r in recs:
            t = tokenize(r["text"])
            Y.append(spans_to_bio(t, r["entities"]))
            P.append(tagger.tag(t))
        return round(seq_f1(Y, P), 4), Y, P
    taggers = {"gazetteer (dictionary)": GazetteerTagger().fit(data["train"]), "crf": CRFTagger("none").fit(data["train"]),
               "crf + gazetteer (full)": CRFTagger("full").fit(data["train"]),
               "crf + gazetteer (value cross-fit)": CRFTagger("crossfit", n_folds=5).fit(data["train"])}
    rows = []
    for name, t in taggers.items():
        rows.append({"model": name, "val": score(t, data["val"])[0], "test": score(t, data["test"])[0],
                     "test_core": score(t, [r for r in data["test"] if not r["is_variant"]])[0],
                     "test_variants": score(t, [r for r in data["test"] if r["is_variant"]])[0]})
        print("   ", rows[-1])
    best = max((n for n in taggers if n.startswith("crf")), key=lambda n: next(r for r in rows if r["model"] == n)["val"])
    _, Y, P = score(taggers[best], data["test"])
    per_type = {k: {m: round(float(v[m]), 3) for m in ("precision", "recall", "f1-score", "support")}
                for k, v in seq_report(Y, P, output_dict=True, zero_division=0).items() if "avg" not in k}
    R["ner"] = {"comparison": rows, "selected": best, "per_type_test": per_type,
                "note": "labels are rule/dictionary-generated (silver); scores measure agreement with them"}
    EntityExtractor(taggers[best]).save(NER_PATH)
    bar(pd.DataFrame(rows), "model", ["val", "test_core", "test_variants"], "NER (span F1, seqeval)", "ner_models.png")
    return taggers[best]


# --------------------------------------------------------------------------- 3. semantic similarity

def similarity(q: dict) -> None:
    print("[3] semantic similarity (paraphrase matching)")
    allq = pd.concat(q.values())
    core = allq[~allq["is_variant"]].reset_index(drop=True)
    var = q["test"][q["test"]["is_variant"]].reset_index(drop=True)
    gold = var["group_id"].map({g: i for i, g in enumerate(core["group_id"])}).to_numpy()
    rows = []

    def evaluate(name, sim):
        rank = (sim > sim[np.arange(len(gold)), gold][:, None]).sum(1) + 1
        rows.append({"method": name, "acc@1": round(float((rank == 1).mean()), 4), "acc@5": round(float((rank <= 5).mean()), 4),
                     "mrr": round(float((1 / rank).mean()), 4)})
        print("   ", rows[-1])
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(core["query"])
    evaluate("tfidf word cosine", (vec.transform(var["query"]) @ vec.transform(core["query"]).T).toarray())
    cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit(core["query"])
    evaluate("tfidf char cosine", (cvec.transform(var["query"]) @ cvec.transform(core["query"]).T).toarray())
    for key in ("minilm", "bge-small"):
        enc = get_encoder(EMBEDDERS[key])
        ec = enc.encode(core["query"].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        ev = enc.encode(var["query"].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        evaluate(f"{key} embedding cosine", ev @ ec.T)
    # Task B (harder): for each held-out core question, the most similar *training* core question should be a
    # different question (from the official NMIMS sources) about the same verified fact (true paraphrase, no shared template).
    tr = q["train"][~q["train"]["is_variant"] & (q["train"]["source_status"] == "VERIFIED")].reset_index(drop=True)
    te = q["test"][~q["test"]["is_variant"] & (q["test"]["source_status"] == "VERIFIED")].reset_index(drop=True)
    tr_f = [set(str(v).split("|")) for v in tr["fact_ids"]]
    te_f = [set(str(v).split("|")) for v in te["fact_ids"]]
    keep = [i for i, fs in enumerate(te_f) if any(fs & t for t in tr_f)]
    te, te_f = te.iloc[keep].reset_index(drop=True), [te_f[i] for i in keep]
    rows_b = []

    def evaluate_b(name, sim):
        top = np.argsort(-sim, axis=1)[:, :5]
        hit = lambda i, k: any(te_f[i] & tr_f[j] for j in top[i, :k])
        rows_b.append({"method": name, "acc@1": round(float(np.mean([hit(i, 1) for i in range(len(te))])), 4),
                       "acc@5": round(float(np.mean([hit(i, 5) for i in range(len(te))])), 4)})
        print("    [same-fact]", rows_b[-1])
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(tr["query"])
    evaluate_b("tfidf word cosine", (v.transform(te["query"]) @ v.transform(tr["query"]).T).toarray())
    for key in ("minilm", "bge-small"):
        enc = get_encoder(EMBEDDERS[key])
        a = enc.encode(te["query"].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        b = enc.encode(tr["query"].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        evaluate_b(f"{key} embedding cosine", a @ b.T)
    R["semantic_similarity"] = {
        "task_a": f"match {len(var)} rule-generated rewordings to their original among {len(core)} core questions",
        "task_a_results": rows,
        "task_b": f"for {len(te)} held-out core questions, retrieve a training question about the same verified fact",
        "task_b_results": rows_b}
    bar(pd.DataFrame(rows_b), "method", ["acc@1", "acc@5"], "Semantic similarity: same-fact question retrieval", "similarity.png")


# --------------------------------------------------------------------------- 4. retrieval

def retrieval_eval_set(q: dict, chunks) -> pd.DataFrame:
    fact_docs: dict[str, set] = {}
    for c in chunks:
        for f in c.fact_ids:
            fact_docs.setdefault(f, set()).add(c.doc_id)
    t = q["test"][(q["test"]["source_status"] == "VERIFIED") & ~q["test"]["multi_intent"]].copy()
    t["relevant"] = t["fact_ids"].map(lambda v: set().union(*[fact_docs.get(f, set()) for f in str(v).split("|")]))
    return t[t["relevant"].map(len) > 0]


def retrieval(q: dict, intent_model) -> HybridRetriever:
    print("[4] retrieval (RAG)")
    chunks = load_chunks()
    ev = retrieval_eval_set(q, chunks)
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks)
    hybrid = HybridRetriever(bm25, dense)
    pred_intent = intent_model.predict(ev["query"])
    services = load_services()
    bundle = joblib.load(SUBINTENT_PATH)
    sm = bundle["model"]
    routed = {}
    for text, pr, it in zip(ev["query"], sm.predict_proba(ev["query"]), pred_intent):
        known = [i for i, c in enumerate(sm.classes_) if c in services]
        allowed = [i for i in known if services[sm.classes_[i]].intent == it]
        pool = allowed if (bundle["strategy"] == "hierarchical" and allowed) else known
        routed[text] = sm.classes_[max(pool, key=lambda i: pr[i])]
    systems = {"bm25": lambda x, i: bm25.search(x, k=50), "dense (minilm)": lambda x, i: dense.search(x, k=50),
               "hybrid (RRF)": lambda x, i: hybrid.search(x, k=50),
               "hybrid + predicted-intent boost": lambda x, i: hybrid.search(x, k=50, intent=i),
               "hybrid + predicted intent & sub-intent boost": lambda x, i: hybrid.search(x, k=50, intent=i, sub_intent=routed[x])}
    rows = []
    for name, fn in systems.items():
        ranks = []
        for text, rel, it in zip(ev["query"], ev["relevant"], pred_intent):
            ids = [h.doc_id for h in fn(text, it)]
            ranks.append(next((r for r, d in enumerate(ids, 1) if d in rel), 999))
        ranks = np.array(ranks)
        core = ~ev["is_variant"].to_numpy()
        rows.append({"retriever": name, "recall@1": round(float((ranks <= 1).mean()), 4), "recall@3": round(float((ranks <= 3).mean()), 4),
                     "recall@5": round(float((ranks <= 5).mean()), 4), "mrr": round(float(np.where(ranks < 999, 1 / ranks, 0).mean()), 4),
                     "recall@3_core": round(float((ranks[core] <= 3).mean()), 4)})
        print("   ", rows[-1])
    R["retrieval"] = {"n_queries": len(ev), "n_passages": len(chunks), "relevance": "passage cites a fact the query is linked to",
                      "comparison": rows}
    bar(pd.DataFrame(rows), "retriever", ["recall@1", "recall@3", "mrr"], "Retrieval over NMIMS passages", "retrieval.png")
    return hybrid


# --------------------------------------------------------------------------- 5. summarization

def summarization(q: dict, hybrid: HybridRetriever, intent_model, sub_model) -> None:
    print("[5] summarization")
    services = load_services()
    t = q["test"][(q["test"]["source_status"] == "VERIFIED") & ~q["test"]["is_variant"] & ~q["test"]["multi_intent"]]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    enc = get_encoder()
    systems = [LeadSummarizer(), TextRankSummarizer(), MMRSummarizer()]
    res = {s.name: [] for s in systems}
    bundle = joblib.load(SUBINTENT_PATH)
    probas = sub_model.predict_proba(t["query"])
    pred_int = intent_model.predict(t["query"])
    routed = []
    for pr, it in zip(probas, pred_int):
        allowed = [i for i, c in enumerate(sub_model.classes_) if c in services and services[c].intent == it]
        pool = allowed if (bundle["strategy"] == "hierarchical" and allowed) else [i for i, c in enumerate(sub_model.classes_) if c in services]
        routed.append(sub_model.classes_[max(pool, key=lambda i: pr[i])])
    for text, ref, sub in zip(t["query"], t["answer_facts"], routed):
        hits = hybrid.search(text, k=3)
        inputs = [h.chunk.text for h in hits] + list(services[sub].facts if sub in services else [])
        for s in systems:
            summ = s.summarize(text, [h.chunk.text for h in hits] if s.name == "lead" else inputs)
            sc = scorer.score(ref, summ)
            e = enc.encode([ref, summ or " "], normalize_embeddings=True, show_progress_bar=False)
            res[s.name].append([sc["rouge1"].fmeasure, sc["rouge2"].fmeasure, sc["rougeL"].fmeasure, float(e[0] @ e[1]),
                                len(summ.split())])
    rows = [{"summarizer": n, **dict(zip(["rouge1", "rouge2", "rougeL", "embedding_similarity", "avg_words"],
                                          [round(float(x), 4) for x in np.mean(v, axis=0)]))} for n, v in res.items()]
    for r in rows:
        print("   ", r)
    R["summarization"] = {"n_queries": len(t), "reference": "gold verified fact(s) linked to the query",
                          "input": "top-3 hybrid passages + facts of the PREDICTED sub-intent's service (lead: passages only)",
                          "comparison": rows}
    bar(pd.DataFrame(rows), "summarizer", ["rouge1", "rougeL", "embedding_similarity"], "Query-focused summarization", "summarization.png")


# --------------------------------------------------------------------------- 6. end-to-end / information extraction

def end_to_end(q: dict, d: dict) -> None:
    print("[6] end-to-end copilot")
    cp = Copilot.from_disk()
    chunks = load_chunks()
    ev = retrieval_eval_set(q, chunks)
    t = q["test"][~q["test"]["is_variant"] & ~q["test"]["multi_intent"] & (q["test"]["source_status"] == "VERIFIED")]
    rel = dict(zip(ev["query"], ev["relevant"]))
    rows, lat = [], []
    for r in t.itertuples():
        o = cp.run(r.query)
        lat.append(o.timings_ms["total"])
        sr = o.service_request or {}
        gold_docs = set(json.loads(r.required_documents))
        pred_docs = set(sr.get("required_documents", []))
        rows.append({
            "intent_ok": o.intent == r.intent, "sub_intent_ok": o.sub_intent == r.sub_intent.split("|")[0],
            "department_ok": sr.get("department") == str(r.department).split("|")[0],
            "fact_supported": bool(set(sr.get("supporting_facts", [])) & set(str(r.fact_ids).split("|"))),
            "docs_jaccard": (len(gold_docs & pred_docs) / len(gold_docs | pred_docs)) if (gold_docs or pred_docs) else 1.0,
            "relevant_in_top3": bool(rel.get(r.query, set()) & {s["doc_id"] for s in o.sources}) if r.query in rel else None,
            "status": sr.get("status", "oos"),
        })
    e = pd.DataFrame(rows)
    oos = d["test_clinc_oos"].sample(200, random_state=0)["text"]
    R["end_to_end"] = {
        "n": len(e), "intent_acc": round(e["intent_ok"].mean(), 4), "sub_intent_acc": round(e["sub_intent_ok"].mean(), 4),
        "department_acc": round(e["department_ok"].mean(), 4), "answer_backed_by_gold_fact": round(e["fact_supported"].mean(), 4),
        "required_documents_jaccard": round(e["docs_jaccard"].mean(), 4),
        "relevant_passage_in_top3": round(e["relevant_in_top3"].dropna().astype(float).mean(), 4),
        "status_counts": e["status"].value_counts().to_dict(),
        "oos_declined_200": round(float(np.mean([cp.run(x).intent == "out_of_scope" for x in oos])), 4),
        "latency_ms_median": round(float(np.median(lat)), 1), "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
    }
    print("   ", R["end_to_end"])
    ex = cp.run("I lost my ID card and my SAP ID is 70022300145, what should I do? It's urgent")
    R["example"] = {"query": ex.query, "intent": ex.intent, "sub_intent": ex.sub_intent, "answer": ex.answer,
                    "entities": ex.entities, "service_request": ex.service_request, "sources": ex.sources}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = load_intent_splits()
    q = queries()
    intent_model, _, sub_model = classification(d)
    ner(q)
    similarity(q)
    hybrid = retrieval(q, intent_model)
    summarization(q, hybrid, intent_model, sub_model)
    end_to_end(q, d)
    R["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"done in {R['runtime_s']} s -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
