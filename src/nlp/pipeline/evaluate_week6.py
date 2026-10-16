"""Week 6 - checks of the reusable NLP component (nlp.component) on held-out data.

  A. confidence      are the confidence bands meaningful? accuracy per intent / routing band on test_core
  B. decisions       what the component tells the caller to do, and how often that is right
  C. contract        every output validates against its schema; size of the JSON
  D. latency         per call type (analyze, intent, entities, retrieve), p50 / p95
  E. example         one full AnalyzeOutput, used in docs/nlp_component.md

Inference only (no training). Run:  python -m nlp.pipeline.evaluate_week6
Outputs: reports/week6/results.json, reports/week6/example_analyze.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from nlp import REPORTS
from nlp.component import AdminNLP, AnalyzeInput, AnalyzeOutput, Options
from nlp.intent.experiments import load as load_splits

OUT = REPORTS / "week6"
EXAMPLE = "I lost my ID card, my roll no is J054. What should I do? It's urgent"
r4 = lambda x: round(float(x), 4)


def run(nlp: AdminNLP, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in df.itertuples():
        o = nlp.analyze(AnalyzeInput(text=t.text, options=Options(include_timings=True)))
        rows.append({"text": t.text, "gold_intent": t.intent, "gold_sub": t.sub_intent, "intent": o.intent.intent,
                     "intent_conf": o.intent.confidence.score, "intent_level": o.intent.confidence.level,
                     "sub": o.routing.sub_intent if o.routing else "out_of_scope",
                     "sub_conf": o.routing.confidence.score if o.routing else None,
                     "sub_level": o.routing.confidence.level if o.routing else None,
                     "action": o.decision.action, "n_entities": len(o.entities),
                     "learned_entity_conf": [e.confidence for e in o.entities if e.confidence is not None],
                     "bytes": len(o.model_dump_json()), "ms": o.timings_ms["total"]})
    return pd.DataFrame(rows)


def bands(r: pd.DataFrame) -> dict:
    ins = r[(r["gold_intent"] != "out_of_scope") & r["sub_level"].notna()]   # in scope and routed
    out = {"intent": [], "routing": []}
    for lvl in ("high", "medium", "low"):
        m = r[r["intent_level"] == lvl]
        if len(m):
            out["intent"].append({"level": lvl, "n": len(m), "share": r4(len(m) / len(r)),
                                  "intent_accuracy": r4((m["intent"] == m["gold_intent"]).mean()),
                                  "mean_confidence": r4(m["intent_conf"].mean())})
    for lvl in ("high", "low"):
        m = ins[ins["sub_level"] == lvl]
        if len(m):
            out["routing"].append({"level": lvl, "n": len(m), "share": r4(len(m) / len(ins)),
                                   "routing_accuracy": r4((m["sub"] == m["gold_sub"]).mean()),
                                   "mean_confidence": r4(m["sub_conf"].mean())})
    return out


def decisions(r: pd.DataFrame) -> list[dict]:
    rows = []
    for action, m in r.groupby("action"):
        ins = m[m["gold_intent"] != "out_of_scope"]
        rows.append({"action": action, "n": len(m), "share": r4(len(m) / len(r)),
                     "gold_out_of_scope": int((m["gold_intent"] == "out_of_scope").sum()),
                     "routing_accuracy_in_scope": r4((ins["sub"] == ins["gold_sub"]).mean())
                     if len(ins) and action != "out_of_scope" else None})   # rejected queries are not routed
    order = ["auto_route", "ask_for_information", "refer_to_staff", "confirm_with_student", "out_of_scope"]
    return sorted(rows, key=lambda x: order.index(x["action"]))


def latency(nlp: AdminNLP, texts: list[str]) -> list[dict]:
    calls = {"analyze": nlp.analyze, "intent": nlp.classify, "entities": nlp.extract_entities, "retrieve": nlp.retrieve}
    rows = []
    for name, fn in calls.items():
        ms = []
        for t in texts:
            t0 = time.perf_counter()
            fn(t)
            ms.append((time.perf_counter() - t0) * 1000)
        rows.append({"call": name, "n": len(ms), "p50_ms": round(float(np.median(ms)), 1),
                     "p95_ms": round(float(np.percentile(ms, 95)), 1)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nlp = AdminNLP()
    d = load_splits()
    test = d["test_core"]
    print(f"[A-C] analysing {len(test)} test_core queries (incl. out-of-scope) with pipeline {nlp.model_info().pipeline}")
    r = run(nlp, test)
    ent = [c for cs in r["learned_entity_conf"] for c in cs]
    R = {"pipeline": nlp.model_info().model_dump(), "test_set": "test_core", "n": len(r),
         "n_in_scope": int((r["gold_intent"] != "out_of_scope").sum()),
         "confidence_bands": bands(r), "decisions": decisions(r),
         "auto_route_precision": r4((r.loc[r["action"] == "auto_route", "sub"] ==
                                     r.loc[r["action"] == "auto_route", "gold_sub"]).mean()),
         "entities": {"per_query_mean": r4(r["n_entities"].mean()), "learned_entities": len(ent),
                      "learned_confidence_mean": r4(np.mean(ent)) if ent else None,
                      "learned_confidence_p10": r4(np.percentile(ent, 10)) if ent else None},
         "contract": {"outputs_validated": len(r), "json_bytes_mean": int(r["bytes"].mean()),
                      "json_bytes_max": int(r["bytes"].max())},
         "pipeline_latency_ms": {"p50": round(float(r["ms"].median()), 1), "p95": round(float(np.percentile(r["ms"], 95)), 1)}}
    for k in ("confidence_bands", "decisions"):
        print("   ", k, json.dumps(R[k]))

    print("[D] latency per call")
    sample = test.sample(min(40, len(test)), random_state=0)["text"].tolist()
    nlp.analyze(sample[0])                               # warm-up (lazy model load)
    R["latency"] = latency(nlp, sample)
    print("   ", R["latency"])

    ex = nlp.analyze(AnalyzeInput(text=EXAMPLE, request_id="example-1"))
    AnalyzeOutput.model_validate_json(ex.model_dump_json())
    (OUT / "example_analyze.json").write_text(ex.model_dump_json(indent=2) + "\n", encoding="utf-8")
    R["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(f"done in {R['runtime_s']} s -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
