"""Phase 3 - expansion and split: add rule-based variants of each core query, remove duplicates, and split by group.

Variants reword the same question in the ways students actually type, using a fixed set of transforms:
  casual       lowercase, no final punctuation, "I" -> "i"
  abbrev       common student abbreviations (attendance->attnd, semester->sem, revaluation->reval, please->pls ...)
  typo         one keyboard-style typo (adjacent swap / dropped letter) in a word of 5+ letters
  formal       "Dear Sir/Madam, <question> Kindly advise."
  polite       a greeting prefix or thanks suffix ("hi, ...", "... pls help")
  hinglish     a checked Hindi-English suffix, only on problem statements ("lost my marksheet, ab kya karu?")
Variants copy every label of their core query (same group_id); they are marked is_variant=True with their
variant_type, and their entities are re-annotated on the new text.

Split: 70/15/15 by group_id (a core query and all of its variants always land in the same split), stratified by
sub-intent, so no rewording of a test question is ever seen in training. test_core.csv holds only the original
test queries from official NMIMS sources for evaluation without variants.

Outputs (data/final/): nmims_admin_queries.csv, train.csv, validation.csv, test.csv, test_core.csv,
nmims_ner_annotations.csv, split_statistics.csv
Run:  python -m dataset.phase3_expand
"""
from __future__ import annotations

import random
import re

import pandas as pd

from dataset.config import FINAL, SEED
from dataset.phase2_annotate import CORE_QUERIES, entities_json, language_style, ner_table

VARIANTS_PER_QUERY = 2           # upper bound; some queries have fewer applicable transforms
SPLIT = (0.70, 0.15, 0.15)

ABBREV = {"attendance": "attnd", "semester": "sem", "examination": "exam", "examinations": "exams",
          "revaluation": "reval", "certificate": "cert", "please": "pls", "documents": "docs", "document": "doc",
          "department": "dept", "information": "info", "registration": "reg", "you": "u", "are": "r", "what": "wat",
          "about": "abt", "because": "coz", "library": "lib", "re-examination": "re exam", "re-exam": "reexam",
          "tomorrow": "tmrw", "message": "msg", "number": "no.", "regarding": "reg",
          "application": "appln", "required": "reqd", "college": "clg", "professor": "prof"}
GREETINGS = ["hi, ", "hello, ", "hey, ", "hi sir, ", "good morning, ", "quick question - ", "sorry to bother, "]
THANKS = [", pls help", ", thanks", " - please guide", ", thank you", ". any help?"]
HINGLISH = [", ab kya karu?", ", kya karna hai?", ", koi batao please", ", help karo", ", kaise solve hoga?"]
QUESTION_WORD = re.compile(r"\b(how|what|when|where|who|whom|which|why|can i|is it|do i|will i|should i)\b", re.I)
PROBLEM_START = re.compile(r"^(i |i'm|im |my |lost|failed|missed|need|forgot|got|was|have|paid|cancelled|applied)", re.I)
QUESTION_START = re.compile(r"^(how|what|when|where|who|whom|which|why|can|could|is|are|do|does|did|will|should|am|may|any)\b", re.I)


def casual(q: str, rng) -> str | None:
    out = re.sub(r"[?.!]+$", "", q).strip().lower()
    out = re.sub(r"\bi'm\b", "im", out)
    return out if out != q else None


def abbrev(q: str, rng) -> str | None:
    words = q.split(" ")
    hits = [i for i, w in enumerate(words) if re.sub(r"[^\w-]", "", w.lower()) in ABBREV]
    if not hits:
        return None
    for i in rng.sample(hits, k=min(len(hits), 2)):
        core = re.sub(r"[^\w-]", "", words[i].lower())
        words[i] = words[i].lower().replace(core, ABBREV[core])
    return " ".join(words)


def typo(q: str, rng) -> str | None:
    words = q.split(" ")
    idx = [i for i, w in enumerate(words) if re.fullmatch(r"[A-Za-z]{5,}[?.,!]?", w)]
    if not idx:
        return None
    i = rng.choice(idx)
    w = words[i]
    tail = w[-1] if not w[-1].isalpha() else ""
    w = w[:-1] if tail else w
    j = rng.randrange(1, len(w) - 2)
    w = (w[:j] + w[j + 1] + w[j] + w[j + 2:]) if rng.random() < 0.5 else (w[:j] + w[j + 1:])
    words[i] = w + tail
    return " ".join(words)


def formal(q: str, rng) -> str | None:
    if (len(q.split()) < 4 or len(q.split()) > 25 or re.match(r"(?i)(dear|respected|hello|hi)\b", q)
            or not (QUESTION_START.match(q) or PROBLEM_START.match(q))):
        return None
    body = re.sub(r"\bi\b", "I", q[0].upper() + q[1:])
    if not re.search(r"[?.!]$", body):
        body += "?" if QUESTION_START.match(q) else "."
    return f"Dear Sir/Madam, {body} Kindly advise."


def polite(q: str, rng) -> str | None:
    if len(q.split()) > 25:
        return None
    if rng.random() < 0.6:
        return rng.choice(GREETINGS) + q[0].lower() + q[1:]
    return re.sub(r"[?.!]+$", "", q) + rng.choice(THANKS)


def hinglish(q: str, rng) -> str | None:
    if (QUESTION_START.match(q) or QUESTION_WORD.search(q) or "?" in q or len(q.split()) > 15
            or len(q.split()) < 2 or not PROBLEM_START.match(q)):
        return None
    return re.sub(r"[.!]+$", "", q) + rng.choice(HINGLISH)


TRANSFORMS = {"casual": casual, "abbrev": abbrev, "typo": typo, "formal": formal, "polite": polite,
              "hinglish": hinglish}


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", q.lower()).strip()


def expand(core: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    core = core.assign(is_variant=False, variant_type="original")
    seen = set(core["query"].map(norm))
    variants = []
    for row in core.to_dict("records"):
        names = list(TRANSFORMS)
        rng.shuffle(names)
        made = 0
        for name in names:
            if made == VARIANTS_PER_QUERY:
                break
            v = TRANSFORMS[name](row["query"], rng)
            if not v or norm(v) in seen:
                continue
            seen.add(norm(v))
            variants.append({**row, "query": v, "is_variant": True, "variant_type": name,
                             "entities": entities_json(v), "language_style": language_style(v)})
            made += 1
    return pd.concat([core, pd.DataFrame(variants)], ignore_index=True)


def split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign whole groups to train/validation/test, stratified by the group's (first) sub-intent."""
    rng = random.Random(SEED)
    groups = df.groupby("group_id").agg(sub=("sub_intent", "first"), n=("query", "size")).reset_index()
    groups["strat"] = groups["sub"].str.split("|").str[0]
    assign, names = {}, ("train", "validation", "test")
    for _, g in groups.groupby("strat"):
        ids = g["group_id"].tolist()
        rng.shuffle(ids)
        sizes = dict(zip(g["group_id"], g["n"]))
        if len(ids) < 3:                       # too few groups to hold out: keep them for training
            assign.update({gid: "train" for gid in ids})
            continue
        total = sum(sizes.values())
        target = {n: total * f for n, f in zip(names, SPLIT)}
        have = dict.fromkeys(names, 0)
        for gid in ids:                        # give each group to the split furthest below its target share
            best = max(names, key=lambda n: (target[n] - have[n]) / target[n])
            assign[gid] = best
            have[best] += sizes[gid]
    return df.assign(split=df["group_id"].map(assign))


def main() -> None:
    core = pd.read_csv(CORE_QUERIES)
    df = split(expand(core))
    df = df.sort_values(["group_id", "is_variant"]).reset_index(drop=True)
    df.insert(0, "query_id", [f"NQ{i:05d}" for i in range(1, len(df) + 1)])
    df.to_csv(FINAL / "nmims_admin_queries.csv", index=False)
    for name in ("train", "validation", "test"):
        df[df["split"] == name].to_csv(FINAL / f"{name}.csv", index=False)
    df[(df["split"] == "test") & ~df["is_variant"]].to_csv(FINAL / "test_core.csv", index=False)
    ner_table(df, "query_id").to_csv(FINAL / "nmims_ner_annotations.csv", index=False)

    stats = (df.assign(intent=df["intent"].str.split("|")).explode("intent")
               .pivot_table(index="intent", columns="split", values="query_id", aggfunc="count", fill_value=0)
               .rename(columns={"train": "train_count", "validation": "validation_count", "test": "test_count"}))
    stats = stats[["train_count", "validation_count", "test_count"]]
    stats.loc["TOTAL"] = stats.sum()
    stats.to_csv(FINAL / "split_statistics.csv")
    (FINAL / "core_queries.csv").unlink(missing_ok=True)
    (FINAL / "core_ner_annotations.csv").unlink(missing_ok=True)

    print(f"{len(core)} core + {int(df['is_variant'].sum())} variants = {len(df)} queries")
    print(df["variant_type"].value_counts().to_string())
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
