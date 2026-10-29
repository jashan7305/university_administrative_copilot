"""Phase 4 - validation and reports.

Checks (a failed check is listed in the report and makes the script exit with status 1):
  missing values, duplicate queries, near-duplicates, invalid intent / sub-intent / department, missing source,
  invalid or unretrieved source URL, contradictory labels, malformed NER spans, train/validation/test leakage
  (by group and by normalised text), unsupported factual claims (every cited fact exists and its verbatim quote
  is found in the extracted source text; UNVERIFIED rows carry no department, answer or next action).

Reports (reports/dataset/): dataset_quality_report.md, intent_taxonomy.md, source_inventory.md
Run:  python -m dataset.phase4_validate
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

import pandas as pd

from dataset.config import (CONTRADICTIONS, DEPARTMENTS, EXTRACTED, FINAL, REPORTS, SEED, SUB_TO_INTENT, TAXONOMY,
                            UNSOURCED_SUB_INTENTS, load_facts)

REQUIRED = ["query_id", "group_id", "query", "intent", "sub_intent", "source_status", "split", "user_type",
            "question_type", "language_style", "ambiguity", "confidence"]
REQUIRED_VERIFIED = ["department", "source_id", "source_url", "source_title", "fact_ids", "next_action", "last_verified"]


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(q).lower()).strip()


def table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def counts(series: pd.Series, name: str) -> pd.DataFrame:
    c = series.value_counts().rename_axis(name).reset_index(name="count")
    c["share"] = (c["count"] / c["count"].sum() * 100).round(1).astype(str) + "%"
    return c


def run_checks(df: pd.DataFrame, ner: pd.DataFrame, inv: pd.DataFrame, facts: dict) -> list[tuple[str, int, str]]:
    checks = []
    add = lambda name, n, detail="": checks.append((name, int(n), detail))
    ver = df[df["source_status"] == "VERIFIED"]
    unv = df[df["source_status"] == "UNVERIFIED"]

    add("missing values (required columns)", df[REQUIRED].isna().sum().sum())
    add("missing values (verified-only columns)", ver[REQUIRED_VERIFIED].isna().sum().sum())
    add("exact duplicate queries", df["query"].duplicated().sum())
    keys = df["query"].map(norm)
    add("near-duplicate queries (same normalised text)", keys.duplicated().sum())
    bad_intent = [r for r in df.itertuples() if any(i not in TAXONOMY for i in r.intent.split("|"))]
    add("invalid intent", len(bad_intent))
    bad_sub = [r for r in df.itertuples() if any(s not in SUB_TO_INTENT for s in r.sub_intent.split("|"))
               or set(SUB_TO_INTENT[s] for s in r.sub_intent.split("|")) != set(r.intent.split("|"))]
    add("invalid sub-intent (unknown or not under its intent)", len(bad_sub))
    alts = [a for v in df["alternative_sub_intents"].dropna() for a in str(v).split("|")]
    add("invalid alternative sub-intents (ambiguous rows)", sum(a not in SUB_TO_INTENT for a in alts))
    canon = set(DEPARTMENTS.values())
    add("invalid department", sum(any(d not in canon for d in str(x).split("|")) for x in ver["department"]))
    add("verified rows without source", ver["source_id"].isna().sum())
    ok_urls = set(inv.loc[inv["http_status"].astype(str) == "200", "url"])
    urls = [u for v in ver["source_url"] for u in str(v).split("|")]
    add("invalid URL format", sum(not re.match(r"^https?://[\w.-]+\.\w+", u) for u in urls))
    add("source URL not retrieved (HTTP != 200)", sum(u not in ok_urls for u in urls))
    contra = df.assign(k=keys).groupby("k")["sub_intent"].nunique()
    add("contradictory labels (same text, different sub-intent)", (contra > 1).sum())

    qtext = dict(zip(df["query_id"], df["query"]))
    bad_span = 0
    for qid, g in ner.groupby("query_id"):
        q, end = qtext[qid], -1
        for s in g.sort_values("start_position").itertuples():
            if not (0 <= s.start_position < s.end_position <= len(q)) or q[s.start_position:s.end_position] != s.entity_text \
                    or s.start_position < end:
                bad_span += 1
            end = s.end_position
    add("malformed or overlapping NER spans", bad_span)
    in_queries = sum(len(json.loads(e)) for e in df["entities"])
    add("NER table / entities column mismatch", abs(in_queries - len(ner)))

    add("group_id in more than one split (leakage)", (df.groupby("group_id")["split"].nunique() > 1).sum())
    add("normalised text in more than one split (leakage)", (df.assign(k=keys).groupby("k")["split"].nunique() > 1).sum())

    cited = {f for v in ver["fact_ids"] for f in str(v).split("|")}
    add("cited fact id not found", len(cited - set(facts)))
    squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    texts = {}
    missing_quotes = []
    for fid in cited & set(facts):
        f = facts[fid]
        texts.setdefault(f["source_id"], squash((EXTRACTED / f"{f['source_id']}.txt").read_text()))
        if squash(f["quote"]) not in texts[f["source_id"]]:
            missing_quotes.append(fid)
    add("unsupported claim: fact quote not found in source text", len(missing_quotes), ", ".join(missing_quotes))
    add("unverified rows carrying an answer / department / next action",
        ((unv["answer_facts"].fillna("") != "") | (unv["department"].fillna("") != "") | (unv["next_action"].fillna("") != "")).sum())
    return checks


def quality_report(df, ner, inv, kb, facts, checks) -> str:
    core = df[~df["is_variant"]]
    explode = lambda col: df[col].dropna().astype(str).str.split("|").explode()
    sp = df.pivot_table(index="split", columns="is_variant", values="query_id", aggfunc="count", fill_value=0)
    sp.columns = ["core" if not c else "variants" for c in sp.columns]
    sp["total"] = sp.sum(axis=1)
    sample_core = core.sample(30, random_state=SEED)[["query_id", "query", "sub_intent"]]
    sample_var = df[df["is_variant"]].sample(30, random_state=SEED)[["query_id", "variant_type", "query", "sub_intent"]]
    failed = [c for c in checks if c[1]]
    n_types = inv["source_type"].value_counts()
    lines = [
        "# NMIMS Administrative Query Dataset - Quality Report", "",
        "Generated by `python -m dataset.phase4_validate`.", "",
        "**How the data was made.** Facts were extracted by the team from official NMIMS sources and each carries a "
        "verbatim quote that this script re-checks against the extracted source text. Core queries were **built from "
        "the official NMIMS sources** - from those verified facts (not collected from students). Variants are **rule-generated** rewordings "
        "of core queries (casual, abbreviation, typo, formal, polite, Hinglish). NER labels are rule-based (silver).", "",
        "## Summary", "",
        table(pd.DataFrame([
            ("Sources in inventory", len(inv)), ("Official NMIMS sources retrieved", int((inv["http_status"].astype(str) == "200").sum())),
            ("PDFs", int(inv["extraction_method"].fillna("").str.startswith("pdf").sum())),
            ("FAQ sources", int(n_types.get("FAQ", 0))), ("Policy sources", int(n_types.get("POLICY", 0))),
            ("Form sources", int(n_types.get("FORM", 0))), ("Verified facts", len(facts)),
            ("Intents", df["intent"].str.split("|").explode().nunique()), ("Sub-intents used", explode("sub_intent").nunique()),
            ("Total records", len(df)), ("Unique queries", df["query"].nunique()),
            ("Core queries (official NMIMS sources)", len(core)), ("Rule-generated variants", int(df["is_variant"].sum())),
            ("NER annotations (spans)", len(ner)), ("Queries with >=1 entity", ner["query_id"].nunique()),
            ("Knowledge-base chunks", len(kb)), ("Unverified records", int((df["source_status"] == "UNVERIFIED").sum())),
            ("Ambiguous records (MEDIUM/HIGH)", int(df["ambiguity"].isin(["MEDIUM", "HIGH"]).sum())),
            ("Multi-intent records", int(df["multi_intent"].sum())),
            ("Exact duplicate queries", int(df["query"].duplicated().sum())),
            ("Near-duplicate queries", int(df["query"].map(norm).duplicated().sum())),
        ], columns=["Item", "Value"])), "",
        "## Validation checks", "", table(pd.DataFrame(checks, columns=["check", "failures", "detail"])), "",
        f"**Result: {'PASS' if not failed else 'FAIL'}** ({len(failed)} failing checks).", "",
        "## Split", "", table(sp.reset_index()), "",
        "Splits are by `group_id`: a core query and all its variants are in the same split. `test_core.csv` holds the "
        "core test queries only.", "",
        "## Intent distribution", "", table(counts(df["intent"].str.split("|").explode(), "intent")), "",
        "## Sub-intent distribution", "", table(counts(explode("sub_intent"), "sub_intent")), "",
        "## Department distribution", "", table(counts(explode("department"), "department")), "",
        "## School distribution", "", table(counts(df["school"].fillna("(unverified)"), "school")), "",
        "## Source distribution (records citing each source)", "", table(counts(explode("source_id"), "source_id").head(30)), "",
        "## Query kind / variant type / user type", "",
        table(counts(df["query_kind"], "query_kind")), "", table(counts(df["variant_type"], "variant_type")), "",
        table(counts(df["user_type"], "user_type")), "",
        "## Language style tags", "", table(counts(df["language_style"].str.split(";").explode(), "tag")), "",
        "## NER entity distribution", "", table(counts(ner["entity_type"], "entity_type")), "",
        "## Ambiguity", "", table(counts(df["ambiguity"], "ambiguity")), "",
        "## Unverified records (no official source found)", "",
        "These sub-intents had no official NMIMS source: " + ", ".join(sorted(UNSOURCED_SUB_INTENTS)) + ". Other "
        "unverified rows ask for specifics the sources do not state (e.g. exact duplicate ID card fee).", "",
        table(df[(df["source_status"] == "UNVERIFIED") & ~df["is_variant"]][["query_id", "query", "sub_intent"]]), "",
        "## Contradictory / duplicate NMIMS sources", "",
        table(pd.DataFrame(CONTRADICTIONS, columns=["topic", "sources", "detail", "resolution"])), "",
        "## Potential problems", "",
        "- Core queries are built from official NMIMS sources, not real student traffic; real-world accuracy will be lower than on this test set.",
        "- About two thirds of the records are rule-generated variants; report results on `test_core.csv` as well as `test.csv`.",
        "- NER labels are dictionary/regex-based (silver); a gold NER test set needs manual correction.",
        "- Time-sensitive facts (calendar dates, fees, committee membership) are flagged `requires_current_verification`.",
        "- Several sources are OCR'd scans (revaluation guide, circulars, academic calendars); those rows have MEDIUM confidence.",
        "- Category sizes follow the source material (scholarships and ID cards have few official facts) and were not forced to balance.", "",
        "## Manual spot check sample (30 core, 30 variants)", "", table(sample_core), "", table(sample_var), "",
    ]
    return "\n".join(lines)


def taxonomy_report(df, facts) -> str:
    fact_n = Counter(f["sub_intent"] for f in facts.values())
    q_n = Counter(s for v in df["sub_intent"] for s in str(v).split("|"))
    rows = [(i, s, fact_n.get(s, 0), q_n.get(s, 0), "no official source" if s in UNSOURCED_SUB_INTENTS else "")
            for i, subs in TAXONOMY.items() for s in subs]
    return "\n".join(["# Intent taxonomy", "", f"{len(TAXONOMY)} intents, {sum(len(v) for v in TAXONOMY.values())} sub-intents.", "",
                      table(pd.DataFrame(rows, columns=["intent", "sub_intent", "facts", "queries", "note"])), "",
                      "## Departments (canonical routing targets)", "",
                      table(pd.DataFrame(list(DEPARTMENTS.items()), columns=["code", "department"]))])


def inventory_report(inv) -> str:
    cols = ["source_id", "title", "source_type", "academic_year", "current_or_historical", "relevance", "facts_cited", "url"]
    return "\n".join(["# Source inventory", "", f"{len(inv)} sources; details in `data/final/source_inventory.csv`.", "",
                      table(counts(inv["source_type"], "source_type")), "", table(inv[cols])])


def main() -> int:
    df = pd.read_csv(FINAL / "nmims_admin_queries.csv")
    ner = pd.read_csv(FINAL / "nmims_ner_annotations.csv")
    inv = pd.read_csv(FINAL / "source_inventory.csv")
    kb = pd.read_csv(FINAL / "nmims_knowledge_base.csv")
    facts = load_facts()
    checks = run_checks(df, ner, inv, facts)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "dataset_quality_report.md").write_text(quality_report(df, ner, inv, kb, facts, checks))
    (REPORTS / "intent_taxonomy.md").write_text(taxonomy_report(df, facts))
    (REPORTS / "source_inventory.md").write_text(inventory_report(inv))
    for name, n, detail in checks:
        print(f"{'FAIL' if n else 'ok  '} {name}: {n} {detail}")
    return 1 if any(n for _, n, _ in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
