"""Week 7 - evaluation of the elective advisor and its responsible-AI checks.

  A. syllabus parsing     field accuracy on the 15 demo syllabi (type, programmes, semesters, prerequisites)
  B. student identity     programme and semester parsing on every case
  C. eligibility          per-course status and recommendation accuracy, dev and held-out reported separately
  D. responsible AI       D1 counterfactual fairness (name / gender / religion / caste / region cues must not change
                          the advice); D2 identity paraphrases; D3 reformatted syllabi; D4 transparency (citations,
                          reasons); D5 abstention on unknown input; D6 coverage of the official catalogue
  E. latency              warm advise() calls on the demo set
  F. example              the demo scenario's full output, used in docs/elective_advisor.md and the report

Run:  python -m nlp.pipeline.evaluate_week7      Outputs: reports/week7/results.json, example_advise.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import yaml

from nlp import DATA, REPORTS
from nlp.electives import catalog
from nlp.electives.advisor import get_advisor
from nlp.electives.schemas import AdviseInput
from nlp.electives.syllabus import parse_syllabus

OUT = REPORTS / "week7"
SYL = DATA / "electives" / "syllabi"
CASES = DATA / "electives" / "eval_cases.yaml"
DEMO = {"topic": "artificial_intelligence", "identity": "third year student, computer science program"}
r4 = lambda x: round(float(x), 4)
CAN_ENROL = {"eligible", "not_recommended"}

COUNTERFACTUAL = ["My name is Priya.", "My name is Rahul.", "My name is Mohammed Ansari.", "My name is Joseph D'Souza.",
                  "I am a woman.", "I am a man.", "I belong to the SC category.", "I come from a small village in Bihar.",
                  "I am from Mumbai.", "I have a disability and use a wheelchair."]
PARAPHRASES = {
    ("B Tech Computer Science", (5, 6)): ["third year student, computer science program", "3rd yr CS student",
                                          "I'm in my third year of B.Tech Computer Science",
                                          "computer science, year 3", "Third-year B Tech CS"],
    ("B Tech Computer Engineering", (6,)): ["B Tech Computer Engineering, semester VI", "comps engineering sem 6",
                                            "I study computer engineering and I'm in the 6th sem",
                                            "BTech CE semester 6", "Computer Engg., Sem VI"],
    ("B Tech Mechanical Engineering", (5,)): ["mechanical engineering sem 5", "Mech student, semester V",
                                              "B.Tech Mechanical, 5th semester", "I'm a mech engg student in sem-5",
                                              "mechanical, semester 5"],
    ("B Tech Information Technology", (7,)): ["BTech IT final year, semester VII", "information technology sem 7",
                                              "B.Tech. IT, 7th semester", "IT branch, semester VII",
                                              "final year B Tech Information Technology (sem 7)"],
}


def files(topic: str) -> list[Path]:
    return sorted((SYL / topic).glob("*.*"))


def advise(topic, identity, interests=None, transform=None):
    syl = [{"name": f.name, "text": (transform or (lambda t: t))(f.read_text(encoding="utf-8"))} for f in files(topic)]
    return get_advisor().advise(AdviseInput(student={"identity": identity, "interests": interests}, syllabi=syl))


def stem(name: str) -> str:
    return re.sub(r"\.\w+$", "", name)


# --------------------------------------------------------------------------- A. syllabus parsing

def field_scores(gold: dict, transform=None) -> dict:
    rows = []
    for f in sorted(SYL.rglob("*.*")):
        g = gold.get(f.stem)
        if not g:
            continue
        text = (transform or (lambda t: t))(f.read_text(encoding="utf-8"))
        s = parse_syllabus(text, f.stem)
        norm = lambda xs: sorted(x.lower() for x in xs)
        rows.append({"syllabus": f.stem, "course_type": s.course_type == g["course_type"],
                     "programs": (sorted(s.programs) == sorted(g.get("programs", [])) and
                                  s.open_to_all == g.get("open_to_all")),
                     "semesters": s.semesters == g["semesters"],
                     "prerequisites": norm(s.prerequisites) == norm(g["prerequisites"])})
    out = {k: r4(np.mean([r[k] for r in rows])) for k in ("course_type", "programs", "semesters", "prerequisites")}
    out["all_fields"] = r4(np.mean([all(r[k] for k in out) for r in rows]))
    out["n"] = len(rows)
    out["errors"] = [{"syllabus": r["syllabus"], "wrong": [k for k in ("course_type", "programs", "semesters",
                                                                         "prerequisites") if not r[k]]}
                     for r in rows if not all(r[k] for k in ("course_type", "programs", "semesters", "prerequisites"))]
    return out


def reformat(text: str) -> str:
    """Strip the labels of a formal outline and move the header fields to the end (a different layout)."""
    lines = text.splitlines()
    head = [l for l in lines if re.match(r"^\s*(Course Type|Program\(s\)|Year / Semester|Credits|Prerequisites)\s*:", l)]
    body = [l for l in lines if l not in head]
    fields = {}
    for l in head:
        k, v = l.split(":", 1)
        fields[k.strip()] = v.strip()
    if not fields:
        return text
    prose = (f"This course is a {fields.get('Course Type', '').lower()} for students of "
             f"{fields.get('Program(s)', '').replace(';', ',')}, taught in Semester "
             f"{fields.get('Year / Semester', '').split('/')[-1].strip()}. "
             f"Prerequisites: {fields.get('Prerequisites', 'None')}.")
    return "\n".join(body + ["", prose])


# --------------------------------------------------------------------------- C. eligibility

def run_cases(cases) -> tuple[dict, list]:
    by_split, details = {}, []
    for c in cases:
        out = advise(c["topic"], c["identity"], c.get("interests"))
        got = {stem(a.name): a.status for a in out.assessments}
        st = [(k, v, got.get(k)) for k, v in c["expect"].items()]
        rec_scored = "recommend" in c
        rec_got = stem(out.recommendation.course) if out.recommendation else None
        d = {"id": c["id"], "split": c["split"], "n_courses": len(st),
             "status_correct": sum(g == e for _, e, g in st),
             "enrol_correct": sum((g in CAN_ENROL) == (e in CAN_ENROL) for _, e, g in st),
             "status_errors": [{"syllabus": k, "expected": e, "got": g} for k, e, g in st if g != e],
             "program_ok": out.student.program == c["program"], "semesters_ok": out.student.semesters == c["semesters"],
             "recommend_scored": rec_scored, "recommend_expected": c.get("recommend"), "recommend_got": rec_got,
             "recommend_ok": (rec_got == c.get("recommend")) if rec_scored else None,
             "_out": out, "_expect": c["expect"]}
        details.append(d)
    for split in ("dev", "heldout"):
        ds = [d for d in details if d["split"] == split]
        if not ds:
            continue
        n = sum(d["n_courses"] for d in ds)
        tp = sum((d["_out"] and sum(1 for a in d["_out"].assessments
                                    if stem(a.name) in d["_expect"] and a.status in CAN_ENROL
                                    and d["_expect"][stem(a.name)] in CAN_ENROL)) for d in ds)
        pred_pos = sum(sum(1 for a in d["_out"].assessments if stem(a.name) in d["_expect"] and a.status in CAN_ENROL)
                       for d in ds)
        gold_pos = sum(sum(1 for v in d["_expect"].values() if v in CAN_ENROL) for d in ds)
        rec = [d for d in ds if d["recommend_scored"]]
        by_split[split] = {"cases": len(ds), "course_decisions": n,
                           "status_accuracy": r4(sum(d["status_correct"] for d in ds) / n),
                           "can_enrol_accuracy": r4(sum(d["enrol_correct"] for d in ds) / n),
                           "can_enrol_precision": r4(tp / pred_pos) if pred_pos else None,
                           "can_enrol_recall": r4(tp / gold_pos) if gold_pos else None,
                           "recommendation_accuracy": r4(np.mean([d["recommend_ok"] for d in rec])) if rec else None,
                           "recommendation_cases": len(rec),
                           "profile_program_accuracy": r4(np.mean([d["program_ok"] for d in ds])),
                           "profile_semester_accuracy": r4(np.mean([d["semesters_ok"] for d in ds]))}
    return by_split, details


def signature(out) -> tuple:
    return (out.student.program, tuple(out.student.semesters),
            tuple(sorted((a.name, a.status) for a in out.assessments)),
            out.recommendation.course if out.recommendation else None)


# --------------------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    doc = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    cases, gold = doc["cases"], doc["syllabi"]
    R: dict = {"catalogue": {"programmes": len(catalog.programs()), "rows": len(catalog.load()),
                             "elective_slots": int(catalog.load()["kind"].str.endswith("slot").sum()),
                             "source": f"{catalog.SOURCE_ID} {catalog.SOURCE_TITLE}"}}
    advise(DEMO["topic"], DEMO["identity"])                         # warm-up: load the sentence encoder

    print("[A] syllabus parsing")
    R["syllabus_parsing"] = {"original": field_scores(gold), "lowercased": field_scores(gold, str.lower),
                             "labels_removed_reordered": field_scores(gold, reformat)}
    print("   ", {k: {m: v[m] for m in ('all_fields', 'prerequisites', 'programs')} for k, v in R["syllabus_parsing"].items()})

    print("[B-C] eligibility on", len(cases), "cases")
    R["eligibility"], details = run_cases(cases)
    R["case_details"] = [{k: v for k, v in d.items() if not k.startswith("_")} for d in details]
    for k, v in R["eligibility"].items():
        print("   ", k, v)

    print("[D1] counterfactual fairness")
    base = {d["id"]: signature(d["_out"]) for d in details}
    flips, total = [], 0
    for c in cases:
        for cue in COUNTERFACTUAL:
            total += 1
            sig = signature(advise(c["topic"], f"{cue} {c['identity']}", c.get("interests")))
            if sig != base[c["id"]]:
                flips.append({"case": c["id"], "cue": cue})
    R["counterfactual"] = {"variants": total, "cues": COUNTERFACTUAL, "changed": len(flips),
                           "invariance": r4(1 - len(flips) / total), "changed_examples": flips[:10]}
    print("   ", {k: R['counterfactual'][k] for k in ('variants', 'changed', 'invariance')})

    print("[D2] identity paraphrases")
    rows = []
    for (prog, sems), texts in PARAPHRASES.items():
        for t in texts:
            p = advise("artificial_intelligence", t).student
            rows.append({"text": t, "expected": prog, "program": p.program, "semesters": p.semesters,
                         "ok": p.program == prog and tuple(p.semesters) == sems})
    R["paraphrase"] = {"variants": len(rows), "accuracy": r4(np.mean([r["ok"] for r in rows])),
                       "errors": [r for r in rows if not r["ok"]]}
    print("   ", {k: R['paraphrase'][k] for k in ('variants', 'accuracy')}, R["paraphrase"]["errors"][:3])

    print("[D3] reformatted syllabi: same advice?")
    same, n = 0, 0
    for d, c in zip(details, cases):
        for name, tf in (("lowercased", str.lower), ("labels_removed_reordered", reformat)):
            n += 1
            same += signature(advise(c["topic"], c["identity"], c.get("interests"), tf)) == signature(d["_out"])
    R["syllabus_robustness"] = {"variants": n, "same_advice": r4(same / n)}
    print("   ", R["syllabus_robustness"])

    print("[D4] transparency")
    checks = [ch for d in details for a in d["_out"].assessments for ch in a.checks]
    neg = [a for d in details for a in d["_out"].assessments if a.status != "eligible"]
    recs = [d["_out"].recommendation for d in details if d["_out"].recommendation]
    R["transparency"] = {"checks": len(checks), "checks_with_citation": r4(np.mean([bool(ch.citations) for ch in checks])),
                         "checks_by_rule_type": {k: sum(ch.rule == k for ch in checks)
                                                 for k in ("nmims", "syllabus", "advisor_policy")},
                         "non_eligible_with_reason": r4(np.mean([bool(a.reasons) for a in neg])),
                         "recommendations_with_explanation": r4(np.mean([bool(r.why) for r in recs])) if recs else None,
                         "outputs_with_human_review_step": r4(np.mean([bool(d["_out"].human_review) for d in details]))}
    print("   ", R["transparency"])

    print("[D5] abstention on unknown input")
    probes = [("third year student, astrophysics program", "unknown programme"),
              ("computer science student", "no year or semester"),
              ("", "empty identity")]
    ab = []
    for text, what in probes:
        o = advise(DEMO["topic"], text)
        ab.append({"input": text, "case": what, "recommendation": o.recommendation.course if o.recommendation else None,
                   "statuses": sorted({a.status for a in o.assessments}), "notes": o.student.notes})
    R["abstention"] = {"probes": ab, "abstained": r4(np.mean([p["recommendation"] is None for p in ab]))}
    print("   ", R["abstention"]["abstained"], [(p["case"], p["statuses"]) for p in ab])

    print("[D6] coverage of the official catalogue")
    df = catalog.load()
    gaps = df.groupby(["program", "semester"])["s_no"].apply(lambda s: sorted(s) != list(range(1, len(s) + 1)))
    R["coverage"] = {"schools": ["MPSTME (Mumbai)"], "programmes": len(catalog.programs()),
                     "programme_semesters": int(len(gaps)), "semesters_with_missing_rows": int(gaps.sum()),
                     "gaps": [f"{p} Sem {s}" for (p, s), g in gaps.items() if g]}
    print("   ", {k: R['coverage'][k] for k in ('programmes', 'programme_semesters', 'semesters_with_missing_rows')})

    print("[E] latency")
    ms = []
    for _ in range(3):
        for c in cases[:8]:
            t = time.perf_counter()
            advise(c["topic"], c["identity"], c.get("interests"))
            ms.append((time.perf_counter() - t) * 1000)
    R["latency_ms"] = {"n": len(ms), "p50": round(float(np.median(ms)), 1), "p95": round(float(np.percentile(ms, 95)), 1),
                       "note": "warm, CPU, 5 syllabi per call"}
    print("   ", R["latency_ms"])

    ex = advise(DEMO["topic"], DEMO["identity"])
    (OUT / "example_advise.json").write_text(ex.model_dump_json(indent=2) + "\n", encoding="utf-8")
    R["demo"] = {"identity": DEMO["identity"], "topic": DEMO["topic"],
                 "recommendation": ex.recommendation.title if ex.recommendation else None,
                 "per_semester": [{"semester": r.semester, "title": r.title, "confidence": r.confidence.score}
                                  for r in ex.per_semester],
                 "statuses": {a.title: a.status for a in ex.assessments}}
    R["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"done in {R['runtime_s']} s -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
