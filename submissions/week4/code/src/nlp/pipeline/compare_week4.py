"""Week 4 - baseline vs improved model comparison, component by component, on the same NMIMS splits.

  A  classification   TF-IDF (Week 2) and frozen-embedding ensemble (Week 3)  ->  SetFit (contrastive fine-tuning)
                      and a fine-tuned transformer trained jointly on 95 sub-intents + out_of_scope
  B  NER              rule-based labeller and CRF  ->  fine-tuned transformer token classifier,
                      scored on a hand-annotated GOLD set (120 test queries) as well as the silver test labels
  C  retrieval        BM25 keyword search  ->  frozen vector retrieval  ->  fine-tuned vector retrieval
                      -> + cross-encoder re-ranking (zero-shot and fine-tuned)
  D  summarization    lead / MMR extractive  ->  LLM-assisted (MMR evidence + instruction-tuned LLM)
  E  end to end       Week 3 copilot (v2) vs improved copilot (v3) on the same held-out questions

Only train/validation data are used for training and model selection; test sets are scored once.
Run:  python -m nlp.pipeline.compare_week4 [--skip-llm]
Outputs: models/week4/ (router, ner, dense, cross_encoder, meta.json), reports/week4/results.json + figures/
"""
from __future__ import annotations

import argparse
import json
import re
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score

from dataset.config import SUB_TO_INTENT
from dataset.phase2_annotate import annotate as rule_annotate
from nlp import DATA, MODELS, REPORTS
from nlp.intent.experiments import load as load_intent_splits
from nlp.intent.finetune import JointIntentAdapter, ProbabilityBlend, SetFitClassifier, TransformerClassifier
from nlp.knowledge.store import load_chunks, load_services
from nlp.ner.extractor import CRFTagger, EntityExtractor, GazetteerTagger, bio_to_entities, tokenize
from nlp.ner.transformer_ner import TransformerNER
from nlp.pipeline.copilot import INTENT_PATH, NER_PATH, SUBINTENT_PATH, V3_DIR, Copilot
from nlp.pipeline.summarize import LeadSummarizer, LLMSummarizer, MMRSummarizer, sentences
from nlp.pipeline.train_eval import queries, retrieval_eval_set
from nlp.retrieval.rerank import CROSS_ENCODER, RerankRetriever, finetune_cross_encoder, finetune_dense
from nlp.retrieval.search import BM25Retriever, DenseRetriever, HybridRetriever, get_encoder

OUT = REPORTS / "week4"
FIG = OUT / "figures"
R: dict = {}
mf1 = lambda y, p: round(f1_score(y, p, average="macro", zero_division=0), 4)
acc = lambda y, p: round(accuracy_score(y, p), 4)


def bar(df, x, cols, title, name, xlim=(0, 1)):
    ax = df.set_index(x)[cols].plot.barh(figsize=(7.5, 0.5 * len(df) + 1.6), width=0.8)
    ax.set_xlim(*xlim), ax.set_title(title), ax.set_ylabel(""), ax.legend(loc="lower right", fontsize=8)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160)
    plt.close()


# --------------------------------------------------------------------------- A. classification

def free_gpu() -> None:
    """Release cached GPU memory between steps (several transformer models are trained in one process)."""
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def classification(d, resume: bool = False):
    print("[A] classification: intent + sub-intent routing")
    if resume and (V3_DIR / "router").exists():           # reuse the models and scores of an interrupted run
        cache = json.loads((OUT / "partial_AB.json").read_text())
        meta = json.loads((V3_DIR / "meta.json").read_text())
        tr = TransformerClassifier.load(V3_DIR / "router")
        v2i, v2s = joblib.load(INTENT_PATH)["model"], joblib.load(SUBINTENT_PATH)["model"]
        blend_sub = ProbabilityBlend([tr, v2s], [meta["w_sub"], 1 - meta["w_sub"]])
        R["classification"] = {"comparison": cache["classification_rows"], "transformer_history": cache["transformer_history"],
                               "oos_threshold": meta["threshold"],
                               "blend_weights": {"sub_intent_transformer": meta["w_sub"], "intent_transformer": meta["w_int"]},
                               "note": "Week 2/3 rows reuse the saved models; improved models are trained here on the same split"}
        print("    resumed from saved models")
        return blend_sub, meta["threshold"]
    sub_of = lambda df: df["sub_intent"]                        # out_of_scope rows carry sub_intent = out_of_scope
    in_scope = lambda df: df[df["intent"] != "out_of_scope"]
    rows, times = [], {}

    def evaluate(name, intent_pred, sub_pred, t):
        row = {"model": name, "train_time_s": t}
        for s in ("test_core", "test", "test_variants"):
            row[f"intent_f1_{s}"] = mf1(d[s]["intent"], intent_pred(d[s]["text"]))
        row["oos_recall"] = acc(d["test_clinc_oos"]["intent"], intent_pred(d["test_clinc_oos"]["text"]))
        for s in ("test_core", "test"):
            df = in_scope(d[s])
            p = sub_pred(df["text"])
            row[f"sub_acc_{s}"], row[f"sub_f1_{s}"] = acc(df["sub_intent"], p), mf1(df["sub_intent"], p)
        rows.append(row)
        print("   ", row)

    # baselines (models trained in Weeks 2-3)
    w2 = joblib.load(MODELS / "intent_baseline.joblib")["model"]
    w2s = joblib.load(MODELS / "subintent_baseline.joblib")["model"]
    evaluate("TF-IDF + LogReg (Week 2)", w2.predict, w2s.predict, None)
    v2i, v2s = joblib.load(INTENT_PATH)["model"], joblib.load(SUBINTENT_PATH)["model"]
    evaluate("TF-IDF + frozen embeddings ensemble (Week 3)", v2i.predict, v2s.predict, None)

    # improved 1: SetFit (contrastive fine-tuning of the sentence encoder) on joint labels
    t0 = time.time()
    setfit = SetFitClassifier().fit(d["train"]["text"], sub_of(d["train"]))
    tsf = round(time.time() - t0, 1)
    sf_int = JointIntentAdapter(setfit, SUB_TO_INTENT)
    evaluate("SetFit: fine-tuned sentence transformer + LogReg", sf_int.predict,
             lambda x: _in_scope_argmax(setfit, x), tsf)

    # improved 2: end-to-end fine-tuned transformer on joint labels (selected on validation macro-F1)
    t0 = time.time()
    tr = TransformerClassifier().fit(d["train"]["text"], sub_of(d["train"]), d["val"]["text"], sub_of(d["val"]))
    ttr = round(time.time() - t0, 1)
    tr_int = JointIntentAdapter(tr, SUB_TO_INTENT)
    evaluate("Fine-tuned transformer (BGE-small, joint sub-intent + OOS)", tr_int.predict,
             lambda x: _in_scope_argmax(tr, x), ttr)
    tr.save(V3_DIR / "router")

    # improved 3 (selected): blend the fine-tuned transformer with the Week 3 models; blend weights chosen on validation
    vin = in_scope(d["val"])
    w_sub = max((0.1, 0.2, 0.3, 0.4, 0.5, 0.7), key=lambda w: acc(vin["sub_intent"], _in_scope_argmax(ProbabilityBlend([tr, v2s], [w, 1 - w]), vin["text"])))
    w_int = max((0.1, 0.2, 0.3, 0.4, 0.5, 0.7), key=lambda w: mf1(d["val"]["intent"], ProbabilityBlend([tr_int, v2i], [w, 1 - w]).predict(d["val"]["text"])))
    blend_sub, blend_int = ProbabilityBlend([tr, v2s], [w_sub, 1 - w_sub]), ProbabilityBlend([tr_int, v2i], [w_int, 1 - w_int])
    evaluate("Fine-tuned transformer + Week 3 ensemble (blend)", blend_int.predict, lambda x: _in_scope_argmax(blend_sub, x), ttr)
    sweep = [(t, mf1(d["val"]["intent"], np.where(blend_int.predict_proba(d["val"]["text"]).max(1) < t, "out_of_scope",
                                                   blend_int.predict(d["val"]["text"])))) for t in np.arange(0, 0.61, 0.05)]
    thr = round(float(max(sweep, key=lambda x: (x[1], -x[0]))[0]), 2)
    (V3_DIR / "meta.json").write_text(json.dumps({"threshold": thr, "w_sub": w_sub, "w_int": w_int}))
    R["classification"] = {"comparison": rows, "transformer_history": tr.history_, "oos_threshold": thr,
                           "blend_weights": {"sub_intent_transformer": w_sub, "intent_transformer": w_int},
                           "note": "Week 2/3 rows reuse the saved models; improved models are trained here on the same split"}
    df = pd.DataFrame(rows)
    df["model"] = df["model"].str.replace(r" \(.*", "", regex=True)
    bar(df, "model", ["intent_f1_test_core", "sub_acc_test_core", "oos_recall"], "Classification: baseline vs improved",
        "classification.png")
    return blend_sub, thr


def _in_scope_argmax(model, texts):
    p = model.predict_proba(texts)
    classes = model.classes_
    mask = np.array([c != "out_of_scope" for c in classes])
    p = np.where(mask, p, -1)
    return classes[p.argmax(1)]


# --------------------------------------------------------------------------- B. NER

def span_scores(pred: list[list[tuple]], gold: list[list[tuple]]) -> dict:
    tp_s = tp_r = n_p = n_g = 0
    for P, G in zip(pred, gold):
        n_p, n_g = n_p + len(P), n_g + len(G)
        tp_s += len(set(P) & set(G))
        tp_r += sum(any(p[2] == g[2] and p[0] < g[1] and g[0] < p[1] for g in G) for p in P)
    f = lambda tp: (2 * tp / (n_p + n_g)) if (n_p + n_g) else 0.0
    return {"strict_p": round(tp_s / max(n_p, 1), 4), "strict_r": round(tp_s / max(n_g, 1), 4), "strict_f1": round(f(tp_s), 4),
            "relaxed_f1": round(f(tp_r), 4)}


def ner(q, resume: bool = False):
    print("[B] NER")
    rec = lambda df: [{"text": t, "is_variant": v, "entities": [{"start": e["start"], "end": e["end"], "label": e["type"], "value": e["text"]}
                                                                 for e in json.loads(ents)]}
                      for t, ents, v in zip(df["query"], df["entities"], df["is_variant"])]
    train, val, test = rec(q["train"]), rec(q["val"]), rec(q["test"])
    gold_doc = yaml.safe_load((DATA / "annotations" / "ner_gold.yaml").read_text())["gold"]
    allq = pd.concat(q.values()).set_index("query_id")
    gold = []
    for qid, spans in gold_doc.items():
        text = allq.loc[qid, "query"]
        gs = [(text.index(str(s)), text.index(str(s)) + len(str(s)), t) for s, t in spans]
        gold.append({"id": qid, "text": text, "spans": gs, "variant": bool(allq.loc[qid, "is_variant"])})

    crf = EntityExtractor.load(NER_PATH).crf
    gaz = GazetteerTagger().fit(train)
    if resume and (V3_DIR / "ner").exists():
        tner, ttime = TransformerNER.load(V3_DIR / "ner"), None
        tner.history = json.loads((OUT / "partial_AB.json").read_text())["ner_history"]
    else:
        t0 = time.time()
        tner = TransformerNER().fit(train, val)
        ttime = round(time.time() - t0, 1)
        tner.save(V3_DIR / "ner")

    def spans_rule(text):
        return [(s["start_position"], s["end_position"], s["entity_type"]) for s in rule_annotate(text)]

    def spans_tagger(tagger):
        def f(text):
            toks = tokenize(text)
            return [(e.start, e.end, e.label) for e in bio_to_entities(text, toks, tagger.tag(toks))]
        return f

    def spans_tner(text):
        return [(e.start, e.end, e.label) for e in tner.extract_learned([text])[0]]
    systems = {"Rule-based labeller (dictionary + regex)": spans_rule, "Gazetteer (dictionary from training data)": spans_tagger(gaz),
               "CRF (Week 3)": spans_tagger(crf), "Fine-tuned transformer NER (Week 4)": spans_tner}
    rows = []
    silver_test = [(r["text"], [(e["start"], e["end"], e["label"]) for e in r["entities"]]) for r in test]
    for name, fn in systems.items():
        row = {"model": name}
        for part, sel in (("gold_all", lambda g: True), ("gold_core", lambda g: not g["variant"]), ("gold_variants", lambda g: g["variant"])):
            gs = [g for g in gold if sel(g)]
            sc = span_scores([fn(g["text"]) for g in gs], [g["spans"] for g in gs])
            row.update({f"{part}_{k}": v for k, v in sc.items()})
        row["silver_test_strict_f1"] = span_scores([fn(t) for t, _ in silver_test], [s for _, s in silver_test])["strict_f1"]
        rows.append(row)
        print("   ", {k: v for k, v in row.items() if "gold_all" in k or k in ("model", "silver_test_strict_f1")})
    R["ner"] = {"gold_set": {"queries": len(gold), "spans": sum(len(g["spans"]) for g in gold)}, "comparison": rows,
                "transformer_history": tner.history, "transformer_train_time_s": ttime,
                "note": "The rule-based labeller produced the silver training labels, so it scores ~1.0 on silver by "
                        "construction; the hand-annotated gold set is the fair comparison."}
    df = pd.DataFrame(rows)
    df["model"] = df["model"].str.replace(r" \(.*", "", regex=True)
    bar(df, "model", ["gold_all_strict_f1", "gold_all_relaxed_f1", "gold_variants_relaxed_f1"], "NER on hand-annotated gold set", "ner.png")
    return tner


# --------------------------------------------------------------------------- C. retrieval

def retrieval(q, router, resume: bool = False):
    print("[C] retrieval")
    cache = json.loads((OUT / "partial_AB.json").read_text()) if (OUT / "partial_AB.json").exists() else {}
    resume = resume and "retrieval_rows" in cache and (V3_DIR / "dense").exists() and (V3_DIR / "cross_encoder").exists()
    chunks = load_chunks()
    by_id = {c.doc_id: c for c in chunks}
    fact_docs = {}
    for c in chunks:
        for f in c.fact_ids:
            fact_docs.setdefault(f, []).append(c)
    # training pairs from TRAIN questions only
    tr = q["train"][(q["train"]["source_status"] == "VERIFIED") & ~q["train"]["multi_intent"]]
    pairs = []
    for text, fids in zip(tr["query"], tr["fact_ids"]):
        rel = list({c.doc_id: c for f in str(fids).split("|") for c in fact_docs.get(f, [])}.values())[:2]
        pairs += [(text, c.indexed_text) for c in rel]
    t0 = time.time()
    if not resume:
        finetune_dense(pairs, V3_DIR / "dense")
    tdense = cache["retrieval_train_time_s"]["dense"] if resume else round(time.time() - t0, 1)
    free_gpu()
    bm25 = BM25Retriever(chunks)
    dense_frozen_minilm = DenseRetriever(chunks)
    dense_frozen_bge = DenseRetriever(chunks, model="BAAI/bge-small-en-v1.5")
    dense_ft = DenseRetriever(chunks, model=str(V3_DIR / "dense"))
    hybrid_v2 = HybridRetriever(bm25, dense_frozen_minilm)
    hybrid_ft = HybridRetriever(bm25, dense_ft)
    # cross-encoder fine-tuning: positives + BM25 hard negatives from TRAIN questions
    ce_pairs = []
    for text, fids in zip(tr["query"], tr["fact_ids"]):
        rel_ids = {c.doc_id for f in str(fids).split("|") for c in fact_docs.get(f, [])}
        if not rel_ids:
            continue
        ce_pairs.append((text, by_id[next(iter(rel_ids))].indexed_text, 1))
        negs = [h for h in bm25.search(text, k=8) if h.doc_id not in rel_ids][:2]
        ce_pairs += [(text, h.chunk.indexed_text, 0) for h in negs]
    t0 = time.time()
    if not resume:
        finetune_cross_encoder(ce_pairs, V3_DIR / "cross_encoder")
    tce = cache["retrieval_train_time_s"]["cross_encoder"] if resume else round(time.time() - t0, 1)
    free_gpu()

    ev = retrieval_eval_set(q, chunks)
    services = load_services()
    pint = JointIntentAdapter(router, SUB_TO_INTENT).predict(ev["query"])
    psub = _in_scope_argmax(router, ev["query"])
    route = {t: (i, s) for t, i, s in zip(ev["query"], pint, psub)}
    if resume:
        rows = cache["retrieval_rows"]
        print("    resumed from saved retrieval results")
    else:
        rows = evaluate_retrievers(ev, route, bm25, dense_frozen_minilm, dense_frozen_bge, dense_ft, hybrid_v2, hybrid_ft)
        cache.update(retrieval_rows=rows, retrieval_train_time_s={"dense": tdense, "cross_encoder": tce})
        (OUT / "partial_AB.json").write_text(json.dumps(cache, indent=2))
    R["retrieval"] = {"n_queries": len(ev), "n_passages": len(chunks), "train_pairs_dense": len(pairs), "train_pairs_cross_encoder": len(ce_pairs),
                      "train_time_s": {"dense": tdense, "cross_encoder": tce}, "comparison": rows,
                      "selected": "Hybrid (fine-tuned vector) + boost"}
    bar(pd.DataFrame(rows), "retriever", ["recall@1", "recall@3", "mrr@20"], "Retrieval: keyword vs vector vs re-ranking", "retrieval.png")
    # the re-ranker scored below the fine-tuned hybrid on this data, so the improved copilot uses the hybrid
    return hybrid_ft, route


def evaluate_retrievers(ev, route, bm25, dense_frozen_minilm, dense_frozen_bge, dense_ft, hybrid_v2, hybrid_ft):
    rr_zero = RerankRetriever(hybrid_ft, CROSS_ENCODER)
    rr_ft = RerankRetriever(hybrid_ft, V3_DIR / "cross_encoder")
    systems = {
        "BM25 keyword search": lambda x: bm25.search(x, k=20),
        "Vector, frozen MiniLM": lambda x: dense_frozen_minilm.search(x, k=20),
        "Vector, frozen BGE-small": lambda x: dense_frozen_bge.search(x, k=20),
        "Vector, fine-tuned BGE-small": lambda x: dense_ft.search(x, k=20),
        "Week 3 v2: hybrid + intent/sub-intent boost": lambda x: hybrid_v2.search(x, k=20, intent=route[x][0], sub_intent=route[x][1]),
        "Hybrid (fine-tuned vector) + boost": lambda x: hybrid_ft.search(x, k=20, intent=route[x][0], sub_intent=route[x][1]),
        "+ cross-encoder re-rank (zero-shot)": lambda x: rr_zero.search(x, k=20, intent=route[x][0], sub_intent=route[x][1]),
        "+ cross-encoder re-rank (fine-tuned)": lambda x: rr_ft.search(x, k=20, intent=route[x][0], sub_intent=route[x][1]),
    }
    rows = []
    core = ~ev["is_variant"].to_numpy()
    for name, fn in systems.items():
        t0 = time.time()
        ranks = np.array([next((r for r, h in enumerate(fn(x), 1) if h.doc_id in rel), 999) for x, rel in zip(ev["query"], ev["relevant"])])
        ms = (time.time() - t0) / len(ev) * 1000
        rows.append({"retriever": name, "recall@1": round(float((ranks <= 1).mean()), 4), "recall@3": round(float((ranks <= 3).mean()), 4),
                     "recall@5": round(float((ranks <= 5).mean()), 4), "mrr@20": round(float(np.where(ranks < 999, 1 / ranks, 0).mean()), 4),
                     "recall@3_core": round(float((ranks[core] <= 3).mean()), 4), "ms_per_query": round(ms, 1)})
        print("   ", rows[-1])
    return rows


# --------------------------------------------------------------------------- D. summarization

NUM = re.compile(r"(?:rs\.?\s?)?\d[\d,.:/]*|[\w.+-]+@[\w-]+\.[\w.]+", re.I)


def summarization(q, retriever, router, skip_llm: bool):
    print("[D] summarization")
    services = load_services()
    t = q["test"][(q["test"]["source_status"] == "VERIFIED") & ~q["test"]["is_variant"] & ~q["test"]["multi_intent"]]
    t = t.sample(min(120, len(t)), random_state=7)
    subs = _in_scope_argmax(router, t["query"])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    enc = get_encoder()
    systems = [LeadSummarizer(), MMRSummarizer()]
    if not skip_llm:
        systems.append(LLMSummarizer())
    res = {s.name: [] for s in systems}
    examples = []
    for i, (text, ref, sub) in enumerate(zip(t["query"], t["answer_facts"], subs)):
        hits = retriever.search(text, k=3, intent=SUB_TO_INTENT.get(sub), sub_intent=sub)
        inputs = [h.chunk.text for h in hits] + list(services[sub].facts if sub in services else [])
        evidence = " ".join(inputs).lower()
        row_ex = {"query": text, "reference": ref}
        for s in systems:
            summ = s.summarize(text, [h.chunk.text for h in hits] if s.name == "lead" else inputs)
            sc = scorer.score(ref, summ)
            e = enc.encode([ref, summ or " "], normalize_embeddings=True, show_progress_bar=False)
            nums = [n.lower().strip(".,") for n in NUM.findall(summ)]
            unsupported = sum(n not in evidence for n in nums)
            res[s.name].append([sc["rouge1"].fmeasure, sc["rouge2"].fmeasure, sc["rougeL"].fmeasure, float(e[0] @ e[1]),
                                len(summ.split()), unsupported / max(len(nums), 1) if nums else 0.0])
            row_ex[s.name] = summ
        if i < 4:
            examples.append(row_ex)
    rows = [{"summarizer": n, **dict(zip(["rouge1", "rouge2", "rougeL", "embedding_similarity", "avg_words", "unsupported_number_rate"],
                                          [round(float(x), 4) for x in np.mean(v, axis=0)]))} for n, v in res.items()]
    for r in rows:
        print("   ", r)
    R["summarization"] = {"n_queries": len(t), "reference": "gold verified fact(s) of the question",
                          "input": "top-3 passages from the fine-tuned hybrid retriever + facts of the predicted service",
                          "llm": "Qwen2.5-1.5B-Instruct (local, greedy decoding) over MMR-selected evidence" if not skip_llm else "skipped",
                          "comparison": rows, "examples": examples}
    bar(pd.DataFrame(rows), "summarizer", ["rouge1", "rougeL", "embedding_similarity"], "Summarization: extractive vs LLM-assisted", "summarization.png")


# --------------------------------------------------------------------------- E. end to end

def end_to_end(q, d):
    print("[E] end to end: Week 3 copilot (v2) vs improved copilot (v3)")
    chunks = load_chunks()
    ev = retrieval_eval_set(q, chunks)
    rel = dict(zip(ev["query"], ev["relevant"]))
    t = q["test"][~q["test"]["is_variant"] & ~q["test"]["multi_intent"] & (q["test"]["source_status"] == "VERIFIED")]
    oos = d["test_clinc_oos"].sample(200, random_state=0)["text"]
    out = {}
    for version in ("v2", "v3"):
        cp = Copilot.from_disk(version)
        rows, lat = [], []
        for r in t.itertuples():
            o = cp.run(r.query)
            lat.append(o.timings_ms["total"])
            sr = o.service_request or {}
            g, p = set(json.loads(r.required_documents)), set(sr.get("required_documents", []))
            rows.append({"intent": o.intent == r.intent, "sub": o.sub_intent == r.sub_intent.split("|")[0],
                         "dept": sr.get("department") == str(r.department).split("|")[0],
                         "docs": (len(g & p) / len(g | p)) if (g or p) else 1.0,
                         "top3": bool(rel[r.query] & {s["doc_id"] for s in o.sources}) if r.query in rel else np.nan})
        e = pd.DataFrame(rows)
        out[version] = {"intent_acc": round(e["intent"].mean(), 4), "sub_intent_acc": round(e["sub"].mean(), 4),
                        "department_acc": round(e["dept"].mean(), 4), "documents_jaccard": round(e["docs"].mean(), 4),
                        "relevant_passage_top3": round(float(e["top3"].dropna().mean()), 4),
                        "oos_declined_200": round(float(np.mean([cp.run(x).intent == "out_of_scope" for x in oos])), 4),
                        "latency_ms_median": round(float(np.median(lat)), 1)}
        print("   ", version, out[version])
        if version == "v3":
            ex = cp.run("I lost my ID card and my SAP ID is 70022300145, what should I do? It's urgent")
            R["example_v3"] = {"query": ex.query, "intent": ex.intent, "sub_intent": ex.sub_intent, "answer": ex.answer,
                               "entities": ex.entities, "service_request": ex.service_request}
    R["end_to_end"] = {"n": len(t), **out}
    df = pd.DataFrame([{"metric": k, "Week 3 (v2)": out["v2"][k], "Week 4 (v3)": out["v3"][k]}
                       for k in ("intent_acc", "sub_intent_acc", "department_acc", "documents_jaccard", "relevant_passage_top3", "oos_declined_200")])
    bar(df, "metric", ["Week 3 (v2)", "Week 4 (v3)"], "End-to-end: Week 3 vs Week 4 pipeline", "end_to_end.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true", help="skip the local LLM summarizer (saves ~15 min)")
    ap.add_argument("--resume", action="store_true", help="reuse router/NER models saved by an interrupted run")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    V3_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d, q = load_intent_splits(), queries()
    router, thr = classification(d, args.resume)
    free_gpu()
    ner(q, args.resume)
    free_gpu()
    retriever, _ = retrieval(q, router, args.resume)
    free_gpu()
    summarization(q, retriever, router, args.skip_llm)
    free_gpu()
    end_to_end(q, d)
    R["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"done in {R['runtime_s']} s")


if __name__ == "__main__":
    main()
